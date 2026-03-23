import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
from .unet_base_blocks import Conv1x3x1, ConvBlock
from .utils import pad_to_power_of_2
from einops import rearrange
## Siamese Unet with Learnable Channel attention

# Importance Weighted Channel Attention
class IWCA(nn.Module):
    def __init__(self, in_channels):
        super(IWCA, self).__init__()
        # no mixing of channel information
        self.c0 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=3, groups=in_channels, padding=1)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.importance_wts = None

    def forward(self, x):
        # Group Convolution
        x_c = self.bn0(F.relu(self.c0(x)))
        x_c = self.bn1(F.relu(self.c1(x_c)))
        # Global Average Pooling
        x_avg = self.global_avg_pool(x_c)
        importance_weights = self.sigmoid(x_avg)
        self.importance_wts = importance_weights
        # Scale the original input
        out = x * importance_weights
        return out


class Down1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = Conv1x3x1(in_channels, 64)
        self.l2 = Conv1x3x1(64, 128)
        self.l3 = Conv1x3x1(128, out_channels)
        
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3, [x1, x2]
    
class UpConcat(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, hsi_feat, msi_feat):
        sx, sy = msi_feat.shape[-2] // hsi_feat.shape[-2], msi_feat.shape[-1] // hsi_feat.shape[-1]
        hsi_feat = F.interpolate(hsi_feat, scale_factor=(sx, sy))
        out = torch.cat([hsi_feat, msi_feat], dim=1)
        return self.bn(F.relu(self.conv(out)))
        
        
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        self.deconv3 = nn.ConvTranspose2d(in_channels*2, 128, 
                                          kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x

    
            
class Up1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        # up 1x3x1
        self.deconv3 = nn.ConvTranspose2d(in_channels, 128, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, z, skip_connection):
        x = self.bn(F.relu(self.conv(z)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x


class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, output_dim, initial_scale=10.0):
        super(GaussianFourierFeatureTransform, self).__init__()
        self.B = torch.randn(input_dim, output_dim)
        self.scale = nn.Parameter(torch.tensor(initial_scale))
        self.pi = 3.14159265359

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, "B C H W -> B (H W) C")
        x_proj = torch.matmul(x, self.B) * self.scale * 2 * self.pi
        x_out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x_out = rearrange(x_out, "B (H W) C -> B C H W", H=H, W=W)
        return x_out

class NeuralFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, output_dim, initial_scale=10.0):
        super(NeuralFourierFeatureTransform, self).__init__()
        self.B = nn.Parameter(torch.randn(input_dim, output_dim))
        self.scale = nn.Parameter(torch.tensor(initial_scale)).to(torch.float)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, "B C H W -> B (H W) C")
        x_proj = torch.matmul(x, self.B) * self.scale
        x_out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x_out = rearrange(x_out, "B (H W) C -> B C H W", H=H, W=W)
        return x_out


class HSIToGTStem(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.refine = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1),
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.refine(x)
        return x


class MSIToGTStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, out_channels, kernel_size=5, stride=5, padding=2),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1)
        )

    def forward(self, x):
        return self.stem(x)


class SiameseEncoder(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, use_nfft=True):
        super().__init__()
        # hsi_enc -> 31 x h x w -> 256  x 1 x 1
        self.use_nfft = use_nfft
        if self.use_nfft:
            self.gff_hsi = NeuralFourierFeatureTransform(hsi_in, hsi_in)
            self.gff_msi = NeuralFourierFeatureTransform(msi_in, msi_in)
            hsi_in = 2 * hsi_in
            msi_in = 2 * msi_in
        self.channel_selector = IWCA(hsi_in)


        self.hsi_to_gt = HSIToGTStem(hsi_in, hsi_in)
        self.msi_to_gt = MSIToGTStem(msi_in, msi_in)

        self.hsi_enc = Down1x3x1(hsi_in, latent_dim)
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        
    def forward(self, hsi, msi):

        B, _, h, w = hsi.shape
        _, _, H, W = msi.shape

        gt_h = h * 2
        gt_w = w * 2

        expected_rgb_h = gt_h * 10
        expected_rgb_w = gt_w * 10


        if h_rgb != expected_rgb_h or w_rgb != expected_rgb_w:
            raise ValueError(
                f"RGB/HSI crop mismatch. "
                f"Given HSI {h_hsi}x{w_hsi}, expected RGB {expected_rgb_h}x{expected_rgb_w}, "
                f"got {h_rgb}x{w_rgb}."
            )

        if self.use_nfft:
            hsi = self.gff_hsi(hsi)
            msi = self.gff_msi(msi)

        hsi = self.channel_selector(hsi)

        hsi_gt = self.hsi_to_gt(hsi)
        msi_gt = self.msi_to_gt(msi)

        z_hsi, hsi_out = self.hsi_enc(hsi_gt)
        z_msi, msi_out = self.msi_enc(msi_gt) # apply bilinear upsample here
        # get scale of upsampling
        # sx, sy = z_msi.shape[-2] // z_hsi.shape[-2], z_msi.shape[-1] // z_hsi.shape[-1]
        # z_hsi = F.interpolate(z_hsi, scale_factor=(sx, sy))
        return z_hsi, z_msi, hsi_out, msi_out


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.upcat2 = UpConcat(latent_dim//2)# [B, 128, 64, 64]
        self.upcat1 = UpConcat(latent_dim//4)# [B, 64, 128, 128]
        self.decoder = Up(latent_dim, out_channels)

    def forward(self, z, hsi_out, msi_out):
        # merge outputs of hsi and msi encoder
        out2 = self.upcat2(hsi_out[1], msi_out[1])
        out1 = self.upcat1(hsi_out[0], msi_out[0])
        x = self.decoder(z, [out1, out2])
        return x


class GRSiameseUNet(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels, **kwargs):
        super().__init__()
        self.encoder = SiameseEncoder(hsi_in, msi_in, latent_dim, use_nfft=False)
        self.decoder = SegmentationDecoder(latent_dim, output_channels)
        
    def forward(self, hsi, msi):
        orig_ht, orig_width = msi.shape[2:]
        # hsi = hsi.to(torch.double)
        # msi = msi.to(torch.double)
        hsi = hsi.float()
        msi = msi.float()
        # msi = pad_to_power_of_2(msi)
        # hsi = pad_to_power_of_2(hsi)
        
        z_hsi, z_msi, hsi_out, msi_out = self.encoder(hsi, msi)
        z = torch.cat([z_hsi, z_msi], dim=1)
        segmentation_map = self.decoder(z, hsi_out, msi_out)  
        outputs = {
            'preds': segmentation_map[:, :, :orig_ht, :orig_width],
            'embeddings': [z_hsi, z_msi]
        }  
        return outputs


if __name__ == '__main__':
    # usage
    model = GRSiameseUNet(31, 3, 256, 5).to(torch.double)  # Assume output channels for segmentation map is 5
    for i in range(1, 5):
        hsi = torch.rand(2, 31, 64*i, 64*i).double()
        msi = torch.rand(2, 3, 256*i, 256*i).double()
        output = model(hsi, msi)
        print(output['preds'].shape)
        
