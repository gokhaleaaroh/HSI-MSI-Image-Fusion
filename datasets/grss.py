# datasets/grss.py

import numpy as np
import spectral as spy
import tifffile
from pathlib import Path
from einops import rearrange
import torch
from .base_dataset import BaseSegmentationDataset, adjust_gamma_hyperspectral
from .contrast_enhancement import contrast_enhancement

NUM_SPECTRAL_BANDS = 48

WAVELENGTHS = np.array([
    374.4, 388.7, 403.1, 417.4, 431.7, 446.1, 460.4, 474.7,
    489.0, 503.4, 517.7, 532.0, 546.3, 560.6, 574.9, 589.2,
    603.6, 617.9, 632.2, 646.5, 660.8, 675.1, 689.4, 703.7,
    718.0, 732.3, 746.6, 760.9, 775.2, 789.5, 803.8, 818.1,
    832.4, 846.7, 861.1, 875.4, 889.7, 904.0, 918.3, 932.6,
    946.9, 961.2, 975.5, 989.8, 1004.2, 1018.5, 1032.8, 1047.1
])

RGB_WAVELENGTHS = np.array([630.0, 532.0, 465.0])

CLASS_NAMES = [
    'Healthy grass', 'Stressed grass', 'Artificial turf',
    'Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water',
    'Residential buildings', 'Non-residential buildings',
    'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares',
    'Highways', 'Railways', 'Paved parking lots',
    'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats'
]

# HSI georeference: pixel (0,0) center at UTM (271460, 3290891), 1m pixel size
HSI_ORIGIN_X, HSI_ORIGIN_Y = 271460.0, 3290891.0
HSI_PX = 1.0

# Training GT georeference: pixel (0,0) upper-left corner at UTM (272056, 3290290), 0.5m pixel size
GT_ORIGIN_X, GT_ORIGIN_Y = 272056.0, 3290290.0
GT_PX = 0.5


def find_closest_bands(wavelengths, target_wavelengths):
    closest_bands = []
    for target in target_wavelengths:
        index = (np.abs(wavelengths - target)).argmin()
        closest_bands.append(index)
    return closest_bands


def get_evenly_spaced_channels(num_bands, top_k):
    return np.linspace(0, num_bands - 1, top_k, dtype=int)


def load_hsi(data_dir):
    data_dir = Path(data_dir)
    hsi_dir = data_dir / "ImageryAndTrainingGT" / "2018IEEE_Contest" / "Phase2" / "FullHSIDataset"
    hdr_path = str(hsi_dir / "20170218_UH_CASI_S4_NAD83.hdr")
    pix_path = str(hsi_dir / "20170218_UH_CASI_S4_NAD83.pix")
    hsi_file = spy.envi.open(hdr_path, pix_path)
    hsi = np.array(hsi_file.load(), dtype=np.float64)
    # Keep only 48 spectral bands (drop NADIR angle and DEM auxiliary bands)
    hsi = hsi[:, :, :NUM_SPECTRAL_BANDS]
    return hsi


def load_training_gt(data_dir):
    data_dir = Path(data_dir)
    gt_path = data_dir / "ImageryAndTrainingGT" / "2018IEEE_Contest" / "Phase2" / "TrainingGT" / "2018_IEEE_GRSS_DFC_GT_TR.tif"
    gt = tifffile.imread(str(gt_path))
    return gt


def align_hsi_and_gt(hsi, gt):
    """Crop HSI and resample GT to the overlapping region at 1m resolution.

    HSI: origin (271460, 3290891), 1m GSD, shape (1202, 4172, 48)
    GT:  origin (272056, 3290290), 0.5m GSD, shape (1202, 4768)

    Returns: (hsi_crop, gt_resampled) at 1m, spatially aligned.
    Result shape: (601, 2385)
    """
    hsi_lines, hsi_samples = hsi.shape[0], hsi.shape[1]
    gt_lines, gt_samples = gt.shape[0], gt.shape[1]

    hsi_x_min = HSI_ORIGIN_X
    hsi_x_max = HSI_ORIGIN_X + (hsi_samples - 1) * HSI_PX
    hsi_y_max = HSI_ORIGIN_Y
    hsi_y_min = HSI_ORIGIN_Y - (hsi_lines - 1) * HSI_PX

    gt_x_min = GT_ORIGIN_X
    gt_x_max = GT_ORIGIN_X + (gt_samples - 1) * GT_PX
    gt_y_max = GT_ORIGIN_Y
    gt_y_min = GT_ORIGIN_Y - (gt_lines - 1) * GT_PX

    overlap_x_min = max(hsi_x_min, gt_x_min)
    overlap_x_max = min(hsi_x_max, gt_x_max)
    overlap_y_min = max(hsi_y_min, gt_y_min)
    overlap_y_max = min(hsi_y_max, gt_y_max)

    hsi_col_start = int(round((overlap_x_min - HSI_ORIGIN_X) / HSI_PX))
    hsi_col_end = int(round((overlap_x_max - HSI_ORIGIN_X) / HSI_PX)) + 1
    hsi_row_start = int(round((HSI_ORIGIN_Y - overlap_y_max) / HSI_PX))
    hsi_row_end = int(round((HSI_ORIGIN_Y - overlap_y_min) / HSI_PX)) + 1

    hsi_crop = hsi[hsi_row_start:hsi_row_end, hsi_col_start:hsi_col_end, :]

    gt_col_start = int(round((overlap_x_min - GT_ORIGIN_X) / GT_PX))
    gt_row_start = int(round((GT_ORIGIN_Y - overlap_y_max) / GT_PX))

    target_h, target_w = hsi_crop.shape[0], hsi_crop.shape[1]
    gt_rows = gt_row_start + np.arange(target_h) * 2
    gt_cols = gt_col_start + np.arange(target_w) * 2
    gt_rows = np.clip(gt_rows, 0, gt_lines - 1).astype(int)
    gt_cols = np.clip(gt_cols, 0, gt_samples - 1).astype(int)
    gt_resampled = gt[np.ix_(gt_rows, gt_cols)]

    return hsi_crop, gt_resampled


def labels_to_onehot(gt_labels, num_classes=20):
    """Convert integer labels (0-20) to one-hot.
    Class 0 (unclassified) maps to all-zeros. Classes 1-20 map to indices 0-19.
    """
    h, w = gt_labels.shape
    onehot = np.zeros((h, w, num_classes), dtype=np.float64)
    for c in range(1, num_classes + 1):
        onehot[:, :, c - 1] = (gt_labels == c).astype(np.float64)
    return onehot


class GRSSDataset(BaseSegmentationDataset):
    def __init__(self, data_dir,
                 rgb_width, rgb_height,
                 hsi_width, hsi_height,
                 top_k,
                 mode="train", transforms=None, split_ratio=0.8, seed=42,
                 channels=None,
                 window_size=5, conductivity=0.95,
                 gamma=0.4, contrast_enhance=True,
                 **kwargs):
        data_dir = Path(data_dir)

        self.colors = [
            'limegreen', 'olive', 'cyan', 'darkgreen', 'lightgreen',
            'saddlebrown', 'blue', 'salmon', 'red', 'gray',
            'lightgray', 'yellow', 'orange', 'darkgray', 'purple',
            'slategray', 'tan', 'gold', 'maroon', 'crimson'
        ]
        self.label_names = CLASS_NAMES
        self.top_k = top_k
        num_classes = len(self.label_names)

        hsi_full = load_hsi(data_dir)
        gt_raw = load_training_gt(data_dir)
        img_sri, gt_labels = align_hsi_and_gt(hsi_full, gt_raw)

        max_val = img_sri.max()
        if max_val > 0:
            img_sri = img_sri / max_val

        gt = labels_to_onehot(gt_labels, num_classes=num_classes)

        if channels is None or channels == 'all':
            if channels == 'all':
                self.channels = list(range(NUM_SPECTRAL_BANDS))
            else:
                self.channels = get_evenly_spaced_channels(
                    NUM_SPECTRAL_BANDS, self.top_k).tolist()
        else:
            self.channels = channels

        img_sri = adjust_gamma_hyperspectral(img_sri, gamma=gamma)

        if contrast_enhance:
            img_sri = contrast_enhancement(
                (img_sri * 255).astype(np.uint8),
                window_size=window_size,
                conductivity=conductivity) / 255.0

        img_rgb = self.get_rgb(img_sri)

        super().__init__(
            img_sri=img_sri, img_rgb=img_rgb, gt=gt,
            rgb_width=rgb_width, rgb_height=rgb_height,
            hsi_width=hsi_width, hsi_height=hsi_height,
            channels=self.channels, top_k=self.top_k,
            mode=mode, transforms=transforms,
            split_ratio=split_ratio, seed=seed, stride=8)

    def get_rgb(self, img_sri):
        closest_bands = find_closest_bands(WAVELENGTHS, RGB_WAVELENGTHS)
        R = img_sri[:, :, closest_bands[0]]
        G = img_sri[:, :, closest_bands[1]]
        B = img_sri[:, :, closest_bands[2]]
        return np.stack((R, G, B), axis=-1)

