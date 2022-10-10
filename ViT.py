import torch
import torch.optim as optim
import torch.nn as nn
import Dataloader
from tqdm import tqdm
from utils import accuracy

import h5py
import numpy as np

from torchvision import transforms
from vit_pytorch import ViT

from main import *

print("started loading dataloaders")
train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders(
        crop_transform=settings.crop_transform, DA_transform=settings.DA_transform, TTA_transform=settings.TTA_transform)
print("dataloaders loaded")

model = ViT(
    image_size = 96,
    patch_size = 12,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Define the loss function and optimizer
criterion = set_loss_function(loss=settings.loss)
optimizer = optim.Adam(model.parameters(), lr=settings.lr)

# move the input and model to GPU for speed if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = settings.num_epochs
if settings.train:
    best_model, val_accuracies = train(model, criterion, optimizer, num_epochs, train_dataloader, valid_dataloader,
                                        settings.modelname, settings.eval_metric)
else:
    model.load_state_dict(torch.load(f'./models/{settings.modelname}.pth'))
    best_model = model
if settings.TTA:
    test_score = TTA_eval_model(best_model, test_dataloader, settings.eval_metric)
else:
    test_score = eval_model(best_model, test_dataloader, settings.eval_metric, test=True)
print(f'test {settings.eval_metric}: {test_score}')