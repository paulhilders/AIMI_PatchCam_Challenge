from Dataloader import getDataLoaders
from ViT import vit_tiny, vit_small, vit_base

import torch
import torch.nn as nn
from tqdm import tqdm

def train_vit(train_loader, model = vit_tiny(), save = "model.pt"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    epochs = 15
    for epoch in tqdm(range(epochs)):
        for data in train_loader:
            images, y = data
            images = images.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, y)
            del images
            loss.backward()
            optimizer.step()
            del loss
            torch.cuda.empty_cache()
    torch.save(model, save)


if __name__ == '__main__':
    train_dataloader, test_dataloader, valid_dataloader = getDataLoaders()
    train_vit(train_dataloader)