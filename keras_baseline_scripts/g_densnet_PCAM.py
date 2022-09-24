'''
Trains a GDenseNet-40-12 model on the PCAM Dataset.
Used for the AI for Medical Imaging course at the University of Amsterdam.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_gcnn.applications.densenetnew import GDenseNet

# from tensorflow.python.client import device_lib

print("===========================================")
print("*** GPU Availability ***")
# print(device_lib.list_local_devices())
print("===========================================")

### Load data (source: https://github.com/basveeling/pcam)
from keras.utils import HDF5Matrix

x_train = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_train_x.h5', 'r')['x']
y_train = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_train_y.h5', 'r')['y'][:, 0, 0, 0]

x_test = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_test_x.h5', 'r')['x']
y_test = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_test_y.h5', 'r')['y'][:, 0, 0, 0]

x_val = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_valid_x.h5', 'r')['x']
y_val = h5py.File('../AIMI_PatchCam_Challenge/pcamv1/camelyonpatch_level_2_split_valid_y.h5', 'r')['y'][:, 0, 0, 0]

### Train and evaluate model

# Confirmed parameters from paper:
batch_size = 64     # Paper reports 64?
# conv_group = 'C4'
conv_group = 'D4'
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

weights_file = 'GDenseNet_PCAM_Weights_Only_D4_{epoch:02d}-{val_loss:.2f}.h5'

img_rows, img_cols = 96, 96
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
    img_rows, img_cols, img_channels)



# Create the model (without loading weights)
model = GDenseNet(mc_dropout=False, padding=padding, nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                  nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth, classes=nb_classes,
                  use_gcnn=use_gcnn, conv_group=conv_group, activation=activation)

print('Model created')
model.summary()

# Compile model
optimizer = Adam(lr=lr)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
print('Finished compiling')


# Train model
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                               cooldown=0, patience=5, min_lr=0.5e-6)
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')

callbacks = [lr_reducer, model_checkpoint]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle='batch',
          callbacks=callbacks,
          validation_data=(x_val, y_val)
          )

# Evaluate model
val_scores = model.evaluate(x_val, y_val, batch_size=batch_size)
print('Val loss : ', val_scores[0])
print('Val accuracy : ', val_scores[1])

test_scores = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test loss : ', test_scores[0])
print('Test accuracy : ', test_scores[1])


