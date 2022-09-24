"""
g_densnet_PCAM_gradCAM.py: Script which takes a trained GDenseNet
    model and applies it to given samples from the validation set, before
    applying the GradCAM interpretability/explainability technique to visualize
    the model behavior.

Used for the AI for Medical Imaging course at the University of Amsterdam.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import h5py
from PIL import Image, ImageFont, ImageDraw, ImageOps
import cv2

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_gcnn.applications.densenetnew import GDenseNet

from keras.models import save_model, load_model, Model

### Load data (source: https://github.com/basveeling/pcam)
from keras.utils import HDF5Matrix

x_train = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_train_x.h5', 'r')['x']
y_train = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_train_y.h5', 'r')['y'][:, 0, 0, 0]

x_test = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_test_x.h5', 'r')['x']
y_test = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_test_y.h5', 'r')['y'][:, 0, 0, 0]

x_val = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_valid_x.h5', 'r')['x']
y_val = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_valid_y.h5', 'r')['y'][:, 0, 0, 0]

# Confirmed parameters from paper:
batch_size = 64     # Paper reports 64?
conv_group = 'D4'
# conv_group = 'C4'
lr = 1e-3
use_gcnn = True

# Parameter values gathered from Github issues and example scripts
depth=10
dropout_rate=0.0
epochs=20
growth_rate=3
img_channels=3
nb_classes=1
nb_dense_block=5
nb_filter=24
padding = 'same'
include_top = True
activation = 'sigmoid'

checkpoint_path = 'GDenseNet_PCAM_Weights_Only_D4_19-0.30.h5'

img_rows, img_cols = 96, 96
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
    img_rows, img_cols, img_channels)

# Create the model and load weights
model = GDenseNet(mc_dropout=False, padding=padding, nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                  nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth, classes=nb_classes,
                  use_gcnn=use_gcnn, conv_group=conv_group, activation=activation)
model.load_weights(checkpoint_path)

print('Model loaded')
model.summary()

# Compile model
optimizer = Adam(lr=lr)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
print('Finished compiling')



### GRADCAM:
# Parameters:
sample_indices = [20, 40, 100, 120, 180, 29, 56, 86, 96, 106] # Indices of the samples we will consider.
apply_relu = True # Perform ReLU on averaged feature maps before normalizing.
save_folder = "gradcam_samples"

combined_image_success = None
combined_image_fail = None
for sample_ind in sample_indices:
    # Retrieve sample, model predictions, and true label.
    sample = x_val[sample_ind:sample_ind+1]
    preds = model.predict(sample)
    labels = y_val[sample_ind:sample_ind+1]

    pred = preds[0][0]
    pred_label = int(pred > 0.5)
    true_label = labels[0]

    # Backpropagate the gradient of the prediction to the last convolutional layer.
    final_layer_name = "dense_4_1_Gconv2D"
    conv_output = model.get_layer(final_layer_name).output

    grads = K.gradients(model.output[:, 0], conv_output)
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function(sample)

    # Calculate the average of the feature maps, weighted per pixel by the
    # backpropagated gradient.
    output = output.squeeze()
    grads_val = np.array(grads_val).squeeze()

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Apply ReLU to averaged feature map:
    if apply_relu:
        cam = np.maximum(0, cam)

    # Resize heatmap to original image size and Normalize image values to between 0 and 255
    cam = cv2.resize(cam, (96, 96), cv2.INTER_LINEAR)
    cam = cv2.normalize(cam, np.zeros((96, 96)), 0, 255, cv2.NORM_MINMAX)
    cam = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)

    sample = np.array(sample.squeeze())

    gradCam_img = sample * 0.6 + cam * 0.4

    img_row = np.concatenate((sample, cam), axis=1)
    img_row = np.concatenate((img_row, gradCam_img), axis=1)

    cv2.imwrite(os.path.join(save_folder, f"GradCAM_sample{sample_ind}" + ".jpg"), img_row)

    success_label = "SUCCESS" if pred_label == true_label else "FAIL"
    print(f"[{success_label}]: Index {sample_ind} Predicted {pred_label}, true label was {true_label}")

    if pred_label == true_label:
        if combined_image_success is None:
            combined_image_success = img_row
        else:
            combined_image_success = np.concatenate((combined_image_success, img_row), axis=0)
    else:
        if combined_image_fail is None:
            combined_image_fail = img_row
        else:
            combined_image_fail = np.concatenate((combined_image_fail, img_row), axis=0)

cv2.imwrite(os.path.join(save_folder, f"GradCAM_combined_success" + ".jpg"), combined_image_success)
cv2.imwrite(os.path.join(save_folder, f"GradCAM_combined_fail" + ".jpg"), combined_image_fail)
