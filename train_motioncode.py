from motion_code.motion_code import MotionCode
from motion_code.data_processing import load_data, process_data_for_motion_codes
import cv2
from datasets.jasper_ridge import input_processing
import numpy as np

img_path = 'data/jasper/jasper_ridge_224.mat'
gt_path = 'data/jasper/jasper_ridge_gt.mat'

start_band = 380; end_band = 2500
rgb_width = 64; rgb_height = 64
hsi_width = 32; hsi_height = 32

img_sri, gt = input_processing(img_path, gt_path)
img_hsi = img_sri#cv2.pyrDown(img_sri, dstsize=(50, 50))

print("hsi shape: ", img_hsi.shape)
img_hsi_reshaped = img_hsi.reshape(-1, img_sri.shape[-1])
print("hsi reshaped: ", img_hsi_reshaped.shape)

print("gt shape: ", gt.shape)
gt_reshaped = gt.reshape(-1, gt.shape[-1])
print("gt reshaped: ", gt_reshaped.shape)

size_each_class = 50
color = 3
colors = ['red', 'green', 'blue', 'orange']
label_names = ['road', 'tree', 'water', 'dirt']
num_classes = 4
indices = None
print(gt_reshaped)
all_labels = np.argmax(gt_reshaped, axis=1)
for c in range(num_classes):
    print('Class', c)
    indices_in_class = np.where(all_labels == c)[0]
    current_choices = np.random.choice(indices_in_class, size=size_each_class)
    if indices is None:
        indices = current_choices
    else:
        indices = np.append(indices, current_choices)
num_series = indices.shape[0]

all_num_series = img_hsi_reshaped.shape[0]
Y_train_all = img_hsi_reshaped.reshape(all_num_series, 1, -1)
Y_train = img_hsi_reshaped[indices, :].reshape(num_series, 1, -1)
labels_train_all = np.argmax(gt_reshaped, axis=1)
labels_train = np.argmax(gt_reshaped[indices, :], axis=1)
print(Y_train.shape, labels_train.shape)

# Then we process the data for motion code model and generate X-variable, which is needed for training.
X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)
X_train_all, Y_train_all, labels_train_all = process_data_for_motion_codes(Y_train_all, labels_train_all)
print(X_train.shape, Y_train.shape, labels_train.shape)
print(X_train_all.shape, Y_train_all.shape, labels_train_all.shape)

model = MotionCode(m=12, Q=1, latent_dim=2, sigma_y=0.1)

print("X_train shape: ", X_train.shape)

model_path = 'motion_code/saved_models/test_model'
model.load(model_path)

from motion_code.utils import plot_motion_codes

plot_motion_codes(X_train, Y_train, test_time_horizon=None, labels=labels_train, label_names=label_names, \
                           model=model, output_dir='motion_code/out/multiple/', additional_data=None)


# Then we train model on the given X_train, Y_train, label_train set and saved it to a file named test_model.
model_path = 'motion_code/saved_models/' + 'test_model'
model.fit(X_train, Y_train, labels_train, model_path)

model = MotionCode(m=12, Q=1, latent_dim=2, sigma_y=0.1)
