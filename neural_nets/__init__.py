from .cnn import CNN
from .ca_siamese_unet import CASiameseUNet
from .unet import UNet
from .sam_siamese_unet import SamSiameseUNet
from .transformer_siamese import CASiameseTransformer
from .pixel_mlp import PixelMLP
from .gr_siamese_unet import GRSiameseUNet

model_factory = {
    'ca_siamese': CASiameseUNet,
    'gr_siamese': GRSiameseUNet,
    'unet': UNet,
    'sam_siamese': SamSiameseUNet,
    'cnn': CNN,
    'siamese_transformer': CASiameseTransformer,
    'pixel_mlp': PixelMLP
}

model_args = {
    'ca_siamese': (6, 3, 256, 4),
    'gr_siamese': (12, 3, 256, 21),
    'unet': (3, 256, 4),
    'sam_siamese': (6, 3, 256, 4),
    'cnn': (3, 4),
    'siamese_transformer': (6, 3, 256, 4),
}
