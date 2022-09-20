import torch
import torch.nn as nn

from Dataloader import getDataLoaders
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import h5py

model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")


if __name__ == '__main__':
    images = h5py.File('pcamv1/camelyonpatch_level_2_split_train_x.h5', 'r')
    # out = model(train_dataloader)
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
    model = ViTModel.from_pretrained('facebook/dino-vits8')
    inputs = feature_extractor(images=images, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state