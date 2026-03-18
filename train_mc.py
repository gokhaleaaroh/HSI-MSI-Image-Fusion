import matplotlib.pyplot as plt
from motion_code.data_processing import process_data_for_motion_codes
from adversity.transforms import apply_augmentation
from motion_code.motion_code import MotionCode
from train_utils.train_utils import parse_args
from datasets import dataset_factory
import numpy as np
import random
import yaml
import os
import pdb

# Set random seeds
random.seed(42)
np.random.seed(42)


args = parse_args()
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
train_dataset = dataset_factory[config['dataset']['name']](
                **config['dataset']['kwargs'], mode="train", 
                transforms=None)
img_hsi = train_dataset.downsample(train_dataset.img_sri)
img_rgb, gt = train_dataset.img_rgb, train_dataset.gt
gt = train_dataset.downsample(gt)
model_name = config['model']['name']
dataset_name = config['dataset']['name']
save_path = f'models/{model_name}_{dataset_name}.pth'
train_dataset = dataset_factory[config['dataset']['name']](
                **config['dataset']['kwargs'], mode="train", 
                transforms=apply_augmentation)
Y_train, labels_train = train_dataset.Y_train.numpy(), np.argmax(train_dataset.labels_train, axis=1)
X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)
model = MotionCode(m=12, Q=1, latent_dim=2, sigma_y=0.1)
# Then we train model on the given X_train, Y_train, label_train set and saved it to a file named test_model.
os.makedirs('motion_code/saved_models', exist_ok=True)
model_path = os.path.join('motion_code/saved_models', dataset_name)
model.fit(X_train, Y_train, labels_train, model_path)
