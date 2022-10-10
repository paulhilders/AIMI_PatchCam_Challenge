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
import timm

from main import *

print("started loading dataloaders")
train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders()
# train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders(transform=settings.transform)
print("dataloaders loaded")

# model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2, img_size=96, drop_rate=0.3)
# print("With drop_rate 0.3")

model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2, img_size=96)
# model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2, img_size=32)

# Define the loss function and optimizer
criterion = set_loss_function(loss=settings.loss)
optimizer = optimizer = optim.Adam(model.parameters(), lr=settings.lr)

# move the input and model to GPU for speed if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = settings.num_epochs
if settings.train:
    best_model, val_accuracies = train(model, optimizer, num_epochs, train_dataloader, valid_dataloader, settings.modelname, settings.eval_metric, criterion=criterion)
else:
    model.load_state_dict(torch.load(f'./models/{settings.modelname}.pth'))
    best_model = model
test_score = eval_model(best_model, test_dataloader, settings.eval_metric, test=True, criterion=criterion)
print(f'test {settings.eval_metric}: {test_score}')
