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
import gc
from pytorch_pretrained_vit import ViT as V

# def accuracy(predictions, targets):
#     """
#     Computes the prediction accuracy, i.e. the average of correct predictions
#     of the network.
#     """
#     acc = 0
#     y = torch.zeros(len(predictions)).to(device)
#     for i in range(len(predictions)):
#         if predictions[i] > 0:
#             y[i] = 1
#     y = y.int()
#     acc = 1*(y == targets.int()).sum().item()
#     accuracy = acc / len(targets)

#     return accuracy

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    """
    acc = 0
    y = torch.zeros(len(predictions)).to(device)
    for i in range(len(predictions)):
        if predictions[i, 1] > predictions[i, 0]:
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
        loss = criterion(outputs.squeeze(), F.one_hot(labels.to(torch.int64), 2).float())
        del inputs
        loss += loss.item()
        acc += accuracy(outputs, labels)
        count += 1

        del loss
        del labels
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    avg_acc = acc / count
    return avg_acc

def train(model,  criterion, optimizer, epochs, train_dataloader, valid_dataloader):
    val_accuracies = []
    best_val_epoch = -1
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        acc = 0.0
        running_loss = 0.0
        count = 0
        model.train()
        for data in tqdm(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(inputs[0])
            inputs = inputs.permute(0, 3, 1, 2) / 255
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)

            # print(outputs.shape, labels.shape, labels)
            loss = criterion(outputs.squeeze(), F.one_hot(labels.to(torch.int64), 2).float())
            running_loss += loss.item()
            del inputs
            loss.backward()

            optimizer.step()
            
            acc += accuracy(outputs, labels)
            count += 1

            del loss
            del labels
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

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
    # dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    # dino_head = ViT.DINOHead(384,num_classes-1)

    model = ViT.VisionTransformer()
    model.load_state_dict(torch.load('./DINO/dino_deitsmall8_pretrain_full_checkpoint.pth'), strict=False)
    print(model)
    # model = V('B_16_imagenet1k', pretrained=True, num_classes=2, image_size=96)

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
    optimizer = optim.Adam(model.parameters())

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