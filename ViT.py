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
train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders()
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
optimizer = optimizer = optim.Adam(model.parameters(), lr=settings.lr)

# move the input and model to GPU for speed if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = settings.num_epochs
if settings.train:
    best_model, val_accuracies = train(model, criterion, optimizer, num_epochs, train_dataloader, valid_dataloader, settings.modelname, settings.eval_metric)
else:
    model.load_state_dict(torch.load(f'./models/{settings.modelname}.pth'))
    best_model = model
test_score = eval_model(best_model, test_dataloader, criterion, settings.eval_metric)
print(f'test {settings.eval_metric}: {test_score}')



# Taken from densenet.py:

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # move the input and model to GPU for speed if available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# val_accuracies = []
# best_val_epoch = -1
# for epoch in range(10):
#     acc = 0.0
#     running_loss = 0.0
#     count = 0
#     for data in tqdm(train_dataloader):
#         inputs, labels = data[0].to(device), data[1].to(device)
#         inputs = inputs.permute(0, 3, 1, 2).byte() / 255
#         labels = labels.squeeze()
#         optimizer.zero_grad()
#         outputs = model(inputs)

#         loss = criterion(outputs, labels)
#         running_loss += loss.item()

#         loss.backward()

#         optimizer.step()
#         acc += accuracy(outputs, labels)
#         count += 1

#     train_loss = running_loss / count
#     train_acc = acc / count
#     print('train_loss: ', running_loss)

#     valid_loss = 0.0
#     count = 0
#     valid_acc = 0.0
#     model.eval()
#     for data in tqdm(valid_dataloader):
#         inputs, labels = data[0].to(device), data[1].to(device)
#         count += 1
#         inputs = inputs.permute(0, 3, 1, 2).byte() / 255
#         labels = labels.squeeze()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         valid_loss += loss.item()
#         valid_acc += accuracy(outputs, labels)

#     valid_acc = acc / count
#     val_accuracies.append(valid_acc)
#     valid_loss = valid_loss / count
#     print('valid_loss: ', valid_loss)
#     print(
#         f"[Epoch {epoch + 1:2d}] Training accuracy: {train_acc * 100.0:05.2f}%, Validation accuracy: {valid_acc * 100.0:05.2f}%")

#     if len(val_accuracies) == 1 or valid_acc > val_accuracies[best_val_epoch]:
#         print(f'Validation accuracy increased({val_accuracies[best_val_epoch] * 100.0:05.2f}%--->{valid_acc * 100.0:05.2f}%) \t Saving The Model')
#         best_val_epoch = epoch

#         # Saving State Dict
#         torch.save(model.state_dict(), './ViT.pth')