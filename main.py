import torch
import torch.optim as optim
import torch.nn as nn
import Dataloader
from tqdm import tqdm
import csv

from utils import accuracy, auc
import settings

from densenet_pytorch import DenseNet


def load_model(modelname, freeze=False):
    NUM_CLASSES = 2
    model = DenseNet.from_pretrained(modelname)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    if freeze:
        for i, child in enumerate(model.children()):
            if i == 0:
                for param in child.parameters():
                    param.requires_grad = False
    return model


def set_loss_function(loss='cross_entropy'):
    if loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        return criterion


def set_optimizer(optimizer='adam', lr=0.0001):
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return optimizer


def preprocess(inputs, labels):
    inputs = inputs.permute(0, 3, 1, 2).byte() / 255
    labels = labels.squeeze()
    return inputs, labels


def eval_model(model, dataloader, eval_function='accuracy', test=False):
    loss = 0.0
    count = 0
    metric = 0.0
    model.eval()
    if test:
        f = open(f'./predictions/{settings.modelname}_predictions.csv', 'w')
        writer = csv.writer(f)
        writer.writerow('case,\tprediction')
        final = []
        running_i = 0
    for data in tqdm(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = preprocess(inputs, labels)

        outputs = model(inputs)
        if test:
            for i, output in enumerate(outputs):
                writer.writerow(f'{running_i+i},\t{max(output)}')
            running_i += len(outputs)
        loss = criterion(outputs, labels)
        loss += loss.item()
        if eval_function == 'accuracy':
            metric += accuracy(outputs, labels)
        elif eval_function == 'auc':
            metric += auc(outputs, labels)
        count += 1
    
    
    avg_metric = metric / count
    return avg_metric


def train(model, criterion, optimizer, num_epochs, train_dataloader, val_dataloader, modelname, eval_metric):
    val_scores = []
    best_val_epoch = -1
    for epoch in range(num_epochs):
        metric = 0.0
        running_loss = 0.0
        count = 0
        for data in tqdm(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs, labels = preprocess(inputs, labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if eval_metric == 'accuracy':
                metric += accuracy(outputs, labels)
            elif eval_metric == 'auc':
                metric += auc(outputs, labels)
            count += 1

        train_loss = running_loss / count
        train_score = metric / count
        print(f'Train loss: {train_loss}')

        valid_metric = eval_model(model, val_dataloader, eval_metric)
        val_scores.append(valid_metric)
        print(
            f"[Epoch {epoch + 1:2d}] Training {eval_metric}: {train_score * 100.0:05.2f}%, Validation accuracy: {valid_metric * 100.0:05.2f}%")

        if len(val_scores) == 1 or valid_metric > val_scores[best_val_epoch]:
            print(
                f'Validation {eval_metric} increased({val_scores[best_val_epoch] * 100.0:05.2f}%--->{valid_metric * 100.0:05.2f}%) \t Saving The Model')
            best_val_epoch = epoch

            # Saving State Dict
            torch.save(model.state_dict(), f'./models/{modelname}.pth')

    model.load_state_dict(torch.load(f'./models/{modelname}.pth'))
    best_model = model
    return best_model, val_scores


if __name__ == '__main__':
    # Load the dataloaders from the dataset
    print("Started loading Dataloaders...")
    train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders()
    print("Dataloaders loaded")

    # Load the model and define the loss function and optimizer
    model = load_model(modelname=settings.model, freeze=settings.freeze)
    criterion = set_loss_function(loss=settings.loss)
    optimizer = set_optimizer(optimizer=settings.optimizer, lr=settings.lr)

    # move the input and model to GPU for speed if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = settings.num_epochs
    if settings.train:
        best_model, val_accuracies = train(model, criterion, optimizer, num_epochs, train_dataloader, valid_dataloader, settings.modelname, settings.eval_metric)
    else:
        model.load_state_dict(torch.load(f'./models/{settings.modelname}.pth'))
        best_model = model
    test_score = eval_model(best_model, test_dataloader, settings.eval_metric, test=True)
    print(f'test {settings.eval_metric}: {test_score}')