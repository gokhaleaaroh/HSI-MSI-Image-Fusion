from .base_dataset import BaseSegmentationDataset, adjust_gamma_hyperspectral
from .contrast_enhancement import contrast_enhancement
from train_utils.motioncode_selection import get_top_channels
import spectral as sp
import numpy as np

def load_envi_data(pix_path, hdr_path):
    img = sp.envi.open(hdr_path, pix_path)
    cube = np.asarray(img.load()) 
    return cube

def labels_to_onehot(gt_labels, num_classes=21):
    """Convert integer labels (0-20) to one-hot.
    Class 0 (unclassified) maps to all-zeros. Classes 1-20 map to indices 1-20.
    """
    h, w = gt_labels.shape
    onehot = np.zeros((h, w, num_classes), dtype=np.float64)
    for c in range(0, num_classes):
        onehot[:, :, c] = (gt_labels == c).astype(np.float64)
    return onehot

CLASS_NAMES = [
    'Unclassified', 'Healthy grass', 'Stressed grass', 'Artificial turf',
    'Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water',
    'Residential buildings', 'Non-residential buildings',
    'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares',
    'Highways', 'Railways', 'Paved parking lots',
    'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats'
]

NUM_SPECTRAL_BANDS = 50

class GRSSDataset(BaseSegmentationDataset):
    def __init__(self, data_dir='./data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/Aligned/',
                 top_k=12,
                 mode="train", transforms=None, split_ratio=0.8, seed=42,
                 channels=None,
                 window_size=5, conductivity=0.95,
                 gamma=0.4, contrast_enhance=True,
                 **kwargs):
        # data_dir = Path(data_dir)

        self.colors = [
            'limegreen', 'olive', 'cyan', 'darkgreen', 'lightgreen',
            'saddlebrown', 'blue', 'salmon', 'red', 'gray',
            'lightgray', 'yellow', 'orange', 'darkgray', 'purple',
            'slategray', 'tan', 'gold', 'maroon', 'crimson', 'magenta'
        ]
        self.label_names = CLASS_NAMES
        self.top_k = top_k
        num_classes = len(self.label_names)

        self.channels = get_top_channels(num_motion=num_classes,
                                         top_k=self.top_k,
                                         dataset_name='grss')

        hsi_path = data_dir + 'hsi_aligned'
        hsi_hdr = data_dir + 'hsi_aligned.hdr'
        img_sri = load_envi_data(hsi_path, hsi_hdr)

        get_path = data_dir + 'gt_aligned'
        gt_hdr = data_dir + 'gt_aligned.hdr'
        gt_labels = load_envi_data(get_path, gt_hdr)
        gt_labels = gt_labels.squeeze()

        max_val = img_sri.max()
        if max_val > 0:
            img_sri = img_sri / max_val

        gt = labels_to_onehot(gt_labels, num_classes=num_classes)

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

grss_dataset = GRSSDataset()
