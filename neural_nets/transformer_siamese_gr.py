import torch.nn as nn
import torch.nn.functional as F
import torch
from .unet_base_blocks import Conv1x3x1
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
        
    def forward(self, msi_feat, hsi_feat):
        # upsample msi features
        sx, sy = msi_feat.shape[-2] // hsi_feat.shape[-2], msi_feat.shape[-1] // hsi_feat.shape[-1]
        hsi_feat = F.interpolate(hsi_feat, scale_factor=(sx, sy))
        out = torch.cat([hsi_feat, msi_feat], dim=1)
        return self.bn(F.relu(self.conv(out)))
        
        
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        # self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(in_channels)
        self.deconv3 = nn.ConvTranspose2d(in_channels, 128, 
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
        
        # x = self.bn(F.relu(self.conv(z)))
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


class SiameseEncoder(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim):
        super().__init__()
        # hsi_enc -> 31 x h x w -> 256  x 1 x 1
        self.channel_selector = IWCA(hsi_in)
        self.hsi_enc = Down1x3x1(hsi_in, latent_dim)
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        
    def forward(self, hsi, msi):
        hsi = self.channel_selector(hsi)
        z_hsi, hsi_out = self.hsi_enc(hsi)
        z_msi, msi_out = self.msi_enc(msi)
        z_hsi = F.interpolate(z_hsi, size=z_msi.shape[-2:],
                              mode='bilinear', align_corners=False)
        return z_hsi, z_msi, hsi_out, msi_out


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.upcat2 = UpConcat(latent_dim//2)# [B, 128, 64, 64]
        self.upcat1 = UpConcat(latent_dim//4)# [B, 64, 128, 128]
        # self.upcat2 = CrossAttentionBlock(
        #     hsi_channels=latent_dim//2, 
        #     msi_channels=latent_dim//2, out_channels=latent_dim//2)
        # self.upcat1 = CrossAttentionBlock(
        #     hsi_channels=latent_dim//4, 
        #     msi_channels=latent_dim//4, out_channels=latent_dim//4)
        self.decoder = Up(latent_dim, out_channels)

    def forward(self, z, hsi_out, msi_out):
        # merge outputs of hsi and msi encoder
        out2 = self.upcat2(msi_out[1], hsi_out[1])
        out1 = self.upcat1(msi_out[0], hsi_out[0])
        x = self.decoder(z, [out1, out2])
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

def fourier_transform(x):
    return torch.fft.fft2(x)

def inverse_fourier_transform(x):
    return torch.fft.ifft2(x)


class ReduceFourierDimLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReduceFourierDimLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [B, P, C, F]
        B, P, C, F = x.shape
        x = x.view(B * P * C, F)  # Flatten dimensions except the last one
        x = self.linear(x)
        x = x.view(B, P, C, -1)  # Reshape back to [B, P, C, reduced_dim]
        return x

class FourierCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(FourierCrossAttention, self).__init__()
        self.input_dim = input_dim
        self.query_real = nn.Linear(input_dim, input_dim)
        self.query_imag = nn.Linear(input_dim, input_dim)
        self.key_real = nn.Linear(input_dim, input_dim)
        self.key_imag = nn.Linear(input_dim, input_dim)
        self.value_real = nn.Linear(input_dim, input_dim)
        self.value_imag = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        b, c, h, w = x.shape

        x_freq = fourier_transform(x)
        y_freq = fourier_transform(y)

        # Reshape [B, C, H, W] -> [B, H*W, C] so each spatial position is a token
        x_real = rearrange(x_freq.real, 'b c h w -> b (h w) c')
        x_imag = rearrange(x_freq.imag, 'b c h w -> b (h w) c')
        y_real = rearrange(y_freq.real, 'b c h w -> b (h w) c')
        y_imag = rearrange(y_freq.imag, 'b c h w -> b (h w) c')

        # Complex-valued linear projections: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        Q_real = self.query_real(x_real) - self.query_imag(x_imag)
        Q_imag = self.query_real(x_imag) + self.query_imag(x_real)

        K_real = self.key_real(y_real) - self.key_imag(y_imag)
        K_imag = self.key_real(y_imag) + self.key_imag(y_real)

        V_real = self.value_real(y_real) - self.value_imag(y_imag)
        V_imag = self.value_real(y_imag) + self.value_imag(y_real)

        attention_scores_real = torch.matmul(Q_real, K_real.transpose(-2, -1)) - torch.matmul(Q_imag, K_imag.transpose(-2, -1))
        attention_scores_imag = torch.matmul(Q_real, K_imag.transpose(-2, -1)) + torch.matmul(Q_imag, K_real.transpose(-2, -1))
        attention_scores = torch.sqrt(attention_scores_real**2 + attention_scores_imag**2 + 1e-8)

        attention_weights = self.softmax(attention_scores)

        out_real = torch.matmul(attention_weights, V_real)
        out_imag = torch.matmul(attention_weights, V_imag)

        # Reshape back to spatial layout and inverse FFT
        out_complex = torch.complex(out_real, out_imag)
        out_complex = rearrange(out_complex, 'b (h w) c -> b c h w', h=h, w=w)
        out = inverse_fourier_transform(out_complex).real

        return out

class CustomTransformerDecoderWithFourier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomTransformerDecoderWithFourier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.cross_attention = FourierCrossAttention(input_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # 3 stages of 2x upsampling (8x total) to recover spatial resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim // 2, input_dim // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim // 4, input_dim // 8,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(input_dim // 8),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(input_dim // 8, num_classes, kernel_size=1)

    def forward(self, encoder_output, target_size):
        b, c, h, w = encoder_output.size()
        n = h * w

        attended = self.cross_attention(encoder_output, encoder_output)
        src = attended.reshape(b, c, n).permute(2, 0, 1)  # (n, b, c)
        tgt = torch.zeros((n, b, c), device=encoder_output.device,
                           dtype=encoder_output.dtype)

        decoder_output = self.transformer_decoder(tgt, src)  # (n, b, c)
        decoder_output = decoder_output.permute(1, 2, 0).reshape(b, c, h, w)

        decoder_output = self.upsample(decoder_output)
        decoder_output = F.interpolate(decoder_output, size=target_size,
                                       mode='bilinear', align_corners=False)

        output = self.seg_head(decoder_output)
        return output
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, hsi_channels, msi_channels, out_channels):
        super(CrossAttentionBlock, self).__init__()
        self.query = nn.Conv2d(hsi_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(msi_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(msi_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z_hsi, z_msi):
        if z_hsi.shape[-2:] != z_msi.shape[-2:]:
            z_msi = F.interpolate(z_msi, size=z_hsi.shape[-2:],
                                  mode='bilinear', align_corners=False)

        batch_size, _, height, width = z_hsi.size()
        
        # Project HSI embeddings to queries
        proj_query = self.query(z_hsi).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # Project MSI embeddings to keys and values
        proj_key = self.key(z_msi).view(batch_size, -1, height * width)
        proj_value = self.value(z_msi).view(batch_size, -1, height * width)

        # Compute attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        # Apply attention map to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)

        # Combine with the original HSI embedding using the gamma parameter
        out = self.gamma * out + z_hsi
        
        return out

class CASiameseTransformerGR(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels, **kwargs):
        super().__init__()
        self.encoder = SiameseEncoder(hsi_in, msi_in, latent_dim)
        self.attention = AttentionBlock(latent_dim)
        self.cross_attention = CrossAttentionBlock(
            hsi_channels=latent_dim, msi_channels=latent_dim, out_channels=latent_dim)
        self.decoder = CustomTransformerDecoderWithFourier(latent_dim,
                                                           num_classes=output_channels)

    def forward(self, hsi, msi):
        # HSI: [B, C_hsi, h, w]  (low-res hyperspectral)
        # MSI: [B, C_msi, 20h, 20w]  (high-res RGB, 20x HSI each dim)
        # Output: [B, classes, 2h, 2w]  (GT resolution, 2x HSI each dim)
        hsi_h, hsi_w = hsi.shape[2:]
        gt_h, gt_w = hsi_h * 2, hsi_w * 2

        # Downsample MSI from native resolution to GT resolution (10x reduction
        # each dim) so the encoder operates at a tractable scale while still
        # giving the MSI branch 2x the spatial extent of HSI.
        msi_down = F.interpolate(msi, size=(gt_h, gt_w),
                                 mode='area')

        hsi = hsi.to(torch.double)
        msi_down = msi_down.to(torch.double)
        hsi = pad_to_power_of_2(hsi)
        msi_down = pad_to_power_of_2(msi_down)

        z_hsi, z_msi, hsi_out, msi_out = self.encoder(hsi, msi_down)

        z_hsi = self.attention(z_hsi)
        z_msi = self.attention(z_msi)
        z = self.cross_attention(z_hsi, z_msi)

        segmentation_map = self.decoder(z, (gt_h, gt_w))

        outputs = {
            'preds': segmentation_map,
            'embeddings': [z_hsi, z_msi]
        }
        return outputs


# if __name__ == '__main__':
#     # HSI at h×w, MSI at 20h×20w, GT (output) at 2h×2w
#     model = CASiameseTransformer(31, 3, 256, 5).double()
#     for i in range(1, 5):
#         h, w = 8 * i, 8 * i
#         hsi = torch.rand(2, 31, h, w)
#         msi = torch.rand(2, 3, 20 * h, 20 * w)
#         output = model(hsi, msi)
#         gt_shape = (2 * h, 2 * w)
#         print(f"HSI: {hsi.shape}, MSI: {msi.shape}, "
#               f"Preds: {output['preds'].shape}, "
#               f"Expected GT: (2, 5, {gt_shape[0]}, {gt_shape[1]})")
