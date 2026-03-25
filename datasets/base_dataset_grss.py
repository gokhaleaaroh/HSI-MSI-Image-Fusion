import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from train_utils.motioncode_selection import get_top_channels
import pdb
from einops import rearrange
import torch
from .utils import psf2otf

RGB = np.array([630.0, 532.0, 465.0])

def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


# lets just use gamma correction
def adjust_gamma_hyperspectral(image, gamma=0.5):
    epsilon = 1e-8
    # Calculate the inverse gamma
    invGamma = 1.0 / gamma
    # Initialize an array to store the gamma-corrected image
    gamma_corrected_image = np.zeros_like(image)
    # Apply gamma correction to each spectral band
    for band in range(image.shape[2]):
        # Normalize the band to [0, 1]
        normalized_band = image[:, :, band] / (np.max(image[:, :, band]) + epsilon)
        # Apply gamma correction
        gamma_corrected_band = np.power(normalized_band, invGamma)
        # Scale back to original range
        gamma_corrected_image[:, :, band] = gamma_corrected_band * np.max(image[:, :, band])
    return gamma_corrected_image

class BaseSegmentationDatasetGRSS(Dataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self,
                 img_sri, img_rgb, gt, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height,
                 data_type="pixel",
                 channels=None, 
                 mode="train", transforms=None, 
                 split_ratio=0.8, seed=42, A=0.8,
                 top_k=6,
                 stride=1, gt_width=0, gt_height=0, **kwargs):
        self.A = A
        self.channels = channels
        self.img_sri, self.img_rgb, self.gt = img_sri, img_rgb, gt
        self.num_classes = self.gt.shape[-1]
        # self.width, self.height = self.gt.shape[0], self.gt.shape[1]
        self.width, self.height = self.img_rgb.shape[0], self.img_rgb.shape[1]
        self.split_ratio = split_ratio
        self.rgb_width, self.rgb_height = rgb_width, rgb_height
        self.hsi_width, self.hsi_height = hsi_width, hsi_height
        self.gt_width, self.gt_height = gt_width, gt_height
        self.factor = self.rgb_width // self.hsi_width
        if mode == 'test_full':    
            # test on full image at once
            self.rgb_width, self.rgb_height = self.width, self.height
            self.hsi_width, self.hsi_height = self.width, self.height
            
        self.top_k = top_k
        self.sizeI = self.rgb_width
        self.sigma = 2
        # Max indices along width/height dimension for subimage extraction.
        self.max_width_index = self.width - self.rgb_width + 1
        self.max_height_index = self.height - self.rgb_height + 1

        # indices = list(range(0, self.max_width_index * self.max_height_index, stride))

        w = img_sri.shape[0]
        h = img_sri.shape[1]
        k = 2*stride

        indices = [(x, y) for y in [*range(0, h-k, k//2), h-k] 
               for x in [*range(0, w-k, k//2), w-k]]

        if mode != 'test_full':
            train_indices, test_indices = train_test_split(
                indices, test_size=(1 - split_ratio + 0.2), random_state=seed
            ) 
            
        self.data_type = data_type
        # we add 20% to adjust for extra pixels seen because of large patch size
        # see actual number of pixels seen in data percentage
        self.mode = mode
        if self.mode == 'train':
            self.indices = train_indices
        elif self.mode == 'test':
            self.indices = test_indices
        else:
            self.indices = indices # test full image
            
        self.transforms = transforms
        
    def __repr__(self,):
        img_ar = np.zeros((self.img_rgb.shape[0], self.img_rgb.shape[1]))
        total_pixels = self.width * self.height
        for width_index, height_index in self.indices:
            img_ar[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height)] = 1.0
        
        data_percentage = (img_ar.sum() / total_pixels) * 100
        return f"dataset contains {data_percentage}% overall data"

    def __len__(self):
        return len(self.indices)
    
    def H_z(self, z: torch.Tensor, factor: float, fft_B: torch.Tensor):
        # f = torch.fft.rfft(z, 2, onesided=False)
        f = torch.fft.fft2(z) # [1, 31, 512, 512]
        f = torch.view_as_real(f) #[1, 31, 512, 512, 2]
        
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            # Hz = torch.fft.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.view_as_complex(M))
            x = Hz[:, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
            # Hz = torch.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.view_as_complex(M))
            x = Hz[:, :, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
       
        return torch.view_as_real(x)[..., 0]
    
    def downsample(self, sub_sri):
        sub_sri = torch.FloatTensor(sub_sri).permute(2, 1, 0)
        sz = [sub_sri.shape[1], sub_sri.shape[2]]
        fft_B, _ = para_setting('gaussian_blur', self.factor, sz, self.sigma)
        self.fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), 
                                torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        sub_hsi = self.H_z(sub_sri, self.factor, self.fft_B).permute(2, 1, 0).numpy()
        return sub_hsi
    
    def __getitem__(self, idx):
        # idx is in range [0, len(indices)] but 
        # self.indices stores actual indices of bigger ds
        
        hsi_width_index, hsi_height_index = self.indices[idx] 
        gt_width_index = hsi_width_index * 2
        gt_height_index = hsi_height_index * 2
        rgb_width_index = hsi_width_index * 20
        rgb_height_index = hsi_height_index * 20

        # width_index = idx // self.max_height_index
        # height_index = idx % self.max_height_index

        # Get RGB instance and ground-truth mask
        sub_rgb = self.img_rgb[rgb_width_index:(rgb_width_index + self.rgb_width), 
                               rgb_height_index:(rgb_height_index + self.rgb_height), :]

        if self.gt_width == 0:
            self.gt_width = self.rgb_width

        if self.gt_height == 0:
            self.gt_height = self.rgb_height

        sub_gt = self.gt[gt_width_index:(gt_width_index + self.gt_width), 
                               gt_height_index:(gt_height_index + self.gt_height), :]
        sub_sri = self.img_sri[hsi_width_index:(hsi_width_index + self.hsi_width), 
                               hsi_height_index:(hsi_height_index + self.hsi_height), self.channels]
        
        if self.transforms is not None:
            sub_hsi, sub_rgb, sub_gt = self.transforms(sub_sri, sub_gt,
                                                       self.get_rgb, 
                                                       self.downsample, 
                                                       A=self.A, 
                                                       channels=self.channels)
        else:
            # sub_hsi = self.downsample(sub_sri)
            sub_hsi = sub_sri
            sub_hsi = np.moveaxis(sub_hsi, 2, 0)
            sub_rgb = np.moveaxis(sub_rgb, 2, 0)
            sub_gt = np.moveaxis(sub_gt, 2, 0)
            
        # if self.data_type == 'pixel':
        #     sub_hsi, sub_rgb, sub_gt = self.make_pixel_data(sub_hsi, sub_rgb, sub_gt)

        return sub_hsi, sub_rgb, sub_gt
    
    def get_rgb(self, img_sri):
        raise NotImplementedError("write logic for extracting RGB from SRI")
    
    
    def get_pixel_coords(self, x, y):
        H, W, C = x.shape
        h, w, c = y.shape
        factor = H / h
        # Step 1: Generate the pixel coordinates
        # Generate a grid of coordinates
        y_coords, x_coords = torch.meshgrid(torch.arange(H), 
                                        torch.arange(W), indexing='ij')
        # Step 2: Flatten the spatial dimensions
        # Reshape x to have shape (C, H * W)
        #x_flattened = x.reshape(-1, C)  # Flatten H and W
        x_coords_flattened = x_coords.flatten() # Flatten coordinates
        y_coords_flattened = y_coords.flatten()
        # Step 3: Concatenate coordinates and values
        # Stack coordinates to form (2, H * W)
        pixel_coordinates_x = torch.stack([x_coords_flattened, y_coords_flattened], dim=1)
        pixel_coordinates_y = (pixel_coordinates_x // factor).to(torch.int)
        # Stack the coordinates and pixel values along the third dimension
        # Resulting shape: (H * W, 2 + C)
        # result = torch.cat([pixel_coordinates, torch.from_numpy(x_flattened)], dim=1)
        return pixel_coordinates_x, pixel_coordinates_y
        
        
    def make_pixel_data(self, img_hsi, img_rgb, gt):
        img_hsi = rearrange(img_hsi, "C W H -> W H C")
        img_rgb = rearrange(img_rgb, "C W H -> W H C")
        gt = rearrange(gt, "C W H -> W H C")
        # img_hsi = torch.from_numpy(img_hsi)
        # img_rgb = torch.from_numpy(img_rgb)
        # gt = torch.from_numpy(gt)
        num_classes = len(self.label_names)
        pc_rgb, pc_hsi = self.get_pixel_coords(img_rgb, img_hsi)
        super_pixels = torch.from_numpy(img_hsi[pc_hsi[:, 0],pc_hsi[:, 1], :])
        pixels = torch.from_numpy(img_rgb.reshape(-1, img_rgb.shape[-1]))
        series = torch.cat([pixels, super_pixels, pc_rgb], dim=1) 
        gt_reshaped = gt.reshape(-1, gt.shape[-1]) 
        return series, img_rgb, gt_reshaped
        
    def build_pixel_wise_dataset(self):
        num_classes = len(self.label_names)
        self.img_hsi = self.downsample(self.img_sri[:, :, self.channels])
        img_rgb, gt = self.img_rgb, self.gt
        pc_rgb, pc_hsi = self.get_pixel_coords(self.img_rgb, self.img_hsi)
        super_pixels = torch.from_numpy(self.img_hsi[pc_hsi[:, 0],pc_hsi[:, 1], :])
        pixels = torch.from_numpy(img_rgb.reshape(-1, img_rgb.shape[-1]))
        series = torch.cat([pixels, super_pixels, pc_rgb], dim=1) 
        # now we select 
        gt_reshaped = gt.reshape(-1, gt.shape[-1])      
        indices = None
        all_labels = np.argmax(gt_reshaped, axis=1)
        total_size = gt_reshaped.shape[0]
        train_data_percent = self.split_ratio
        train_data_size = int(total_size * train_data_percent)
        size_each_class = train_data_size // num_classes
        for c in range(num_classes):
            indices_in_class = np.where(all_labels == c)[0]
            c_size = min(len(indices_in_class), size_each_class)
            current_choices = np.random.choice(indices_in_class, size=c_size)
            if indices is None:
                indices = current_choices
            else:
                indices = np.append(indices, current_choices)
        num_series = indices.shape[0]
        all_num_series = series.shape[0]
        self.Y_train = series[indices, :].reshape(num_series, -1)
        self.Y_all = series.reshape(all_num_series, -1)
        self.labels_all = gt_reshaped
        self.labels_train = gt_reshaped[indices, :]
