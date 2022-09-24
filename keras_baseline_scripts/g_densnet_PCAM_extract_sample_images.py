"""
g_densnet_PCAM_extract_sample_images.py: Script which takes a trained GDenseNet
    model and applies it to the validation set, in order to extract interesting
    examples. These samples, categorized into groups of "TOUGH", "EASY", and
    "MISLEADING" images, can be used to analyze the models behavior.

Used for the AI for Medical Imaging course at the University of Amsterdam.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import h5py
from PIL import Image, ImageFont, ImageDraw, ImageOps

from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_gcnn.applications.densenetnew import GDenseNet

from keras.models import save_model, load_model

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


nb_samples = 30000 # Number of samples which we will search through.
nb_images = 25
preds = model.predict(x_val[:nb_samples])
labels = y_val[:nb_samples]

easy_samples = []
tough_samples = []
misleading_samples = []
for i in range(nb_samples):
    pred = preds[i][0]
    pred_label = int(pred > 0.5)
    true_label = labels[i]

    if pred_label == true_label and (pred > 0.99 or pred < 0.01):
        # If the prediction was correct with high confidence, we count the sample
        # as "easy"
        if len(easy_samples) < nb_images:
            easy_samples.append((i, pred_label, true_label))
    elif pred_label != true_label and pred > 0.45 and pred < 0.55:
        # If the prediction was incorrect but only with minor margin, we count
        # the sample as "tough"
        if len(tough_samples) < nb_images:
            tough_samples.append((i, pred_label, true_label))
    elif pred_label != true_label and (pred > 0.99 or pred < 0.01):
        # If the prediction was incorrect with high confidence, we count the
        # sample as "misleading".
        if len(misleading_samples) < nb_images:
            misleading_samples.append((i, pred_label, true_label))

with open('validation_example_indices.pkl', 'wb') as f:
    # Save the (index, pred_label, true_label) tuples for each of the categories
    # in a pickle file.
    pickle.dump([easy_samples, tough_samples, misleading_samples], f)

