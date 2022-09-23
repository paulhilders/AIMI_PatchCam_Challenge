import torch
import torch.optim as optim
import torch.nn as nn
import Dataloader
from tqdm import tqdm
# from utils import accuracy
import h5py
import ViT

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    """
    acc = 0
    y = torch.zeros(len(predictions))
    for i in range(len(predictions)):
        if predictions[i] > 0:
            y[i] = 1
    y = y.int()
    acc = 1*(y == targets.int()).sum().item()
    accuracy = acc / len(targets)

    return accuracy

def eval_model(model, dataloader):
    loss = 0.0
    count = 0
    acc = 0.0
    model.eval()
    for data in tqdm(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.permute(0, 3, 1, 2) / 255
        labels = labels.squeeze().float()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss += loss.item()
        acc += accuracy(outputs, labels)
        count += 1

    avg_acc = acc / count
    return avg_acc

def train(model,  criterion, optimizer, epochs, train_dataloader, valid_dataloader):
    val_accuracies = []
    best_val_epoch = -1
    for epoch in range(epochs):
        acc = 0.0
        running_loss = 0.0
        count = 0
        model.train()
        for data in tqdm(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(inputs[0])
            inputs = inputs.permute(0, 3, 1, 2) / 255
            labels = labels.squeeze().float()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()

            loss.backward()

            optimizer.step()
            acc += accuracy(outputs, labels)
            count += 1

        train_loss = running_loss / count
        train_acc = acc / count
        print(f'Train loss: {train_loss}')
        
        valid_acc = eval_model(model, valid_dataloader)
        val_accuracies.append(valid_acc)

        print(
            f"[Epoch {epoch + 1:2d}] Training accuracy: {train_acc * 100.0:05.2f}%, Validation accuracy: {valid_acc * 100.0:05.2f}%")

        if len(val_accuracies) == 1 or valid_acc > val_accuracies[best_val_epoch]:
            print(f'Validation accuracy increased({val_accuracies[best_val_epoch] * 100.0:05.2f}%--->{valid_acc * 100.0:05.2f}%) \t Saving The Model')
            best_val_epoch = epoch
            
            # Saving State Dict
            torch.save(model.state_dict(), './dino_finetuned.pth')
    best_model = torch.load('./dino_finetuned.pth')
    return best_model, val_accuracies


if __name__ == '__main__':
    # Load the dataloaders from the dataset
    print("Started loading Dataloaders...")
    train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders()
    print("Dataloaders loaded")
    num_classes = 2
    training = True

    # Load the model and define the loss function and optimizer
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    dino_head = ViT.DINOHead(384,num_classes-1)

    model = nn.Sequential(dino, dino_head)
    print(model)

    # warmup_teacher_tmp = 0.04
    # teacher_tmp = 0.04
    # warmup_teacher_temp_epochs = 0
    epochs = 100
    # criterion = ViT.DINOLoss(
    #     100,
    #     2,  # total number of crops = 2 global crops + local_crops_number
    #     warmup_teacher_tmp,
    #     teacher_tmp,
    #     warmup_teacher_temp_epochs,
    #     epochs,
    # )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # move the input and model to GPU for speed if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if training:
        best_model, val_accuracies = train(model, criterion, optimizer, epochs, train_dataloader, valid_dataloader)
    else:
        model.load_state_dict(torch.load('./dino_finetuned.pth'))
        best_model = model
    test_acc = eval_model(best_model, test_dataloader)
    print(f'test accuracy: {test_acc}')