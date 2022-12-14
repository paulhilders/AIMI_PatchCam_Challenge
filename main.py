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
    # If you want Densenet, if not comment out line 17 and 18:
    model = DenseNet.from_pretrained(modelname)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    # If you want ViT uncomment following line:
    #model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2, img_size=96)
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


def eval_model(model, dataloader, eval_function='accuracy', test=False, criterion=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    count = 0
    metric = 0.0
    model.eval()
    with torch.no_grad():
        if test:
            f = open(f'./predictions/{settings.modelname}_predictions.csv', 'w')
            writer = csv.writer(f)
            writer.writerow(['case','prediction'])
            running_i = 0
        for i_batch, sample_batched in tqdm(enumerate(dataloader)):
            inputs, labels = sample_batched['image'].to(device), sample_batched['label'].to(device)

            outputs = model(inputs)
            if test:
                sigmoid = nn.Sigmoid()
                for i, output in enumerate(outputs):
                    probs = sigmoid(output)
                    writer.writerow([f'{running_i+i}',f'{probs[1]}'])
                running_i += len(outputs)
            loss = criterion(outputs, labels)
            if eval_function == 'accuracy':
                metric += accuracy(outputs, labels)
            elif eval_function == 'auc':
                metric += auc(outputs, labels)
            count += 1

    avg_metric = metric / count
    return avg_metric


def TTA_eval_model(model, dataloader, eval_function='accuracy'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    count = 0
    metric = 0.0
    model.eval()

    f = open(f'./predictions/{settings.modelname}_TTA_predictions.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['case', 'prediction'])

    n = 10 # number of times we do prediction for one image
    final_predictions = None
    for j in range(n):
        temp_predictions = None
        for image_id, sample in tqdm(enumerate(dataloader)):
            image, label = sample['image'].to(device), sample['label'].to(device)

            with torch.no_grad():
                output = model(image)

            if eval_function == 'accuracy':
                metric += accuracy(output, label)
            elif eval_function == 'auc':
                metric += auc(output, label)
            count += 1
            for p in output:
                if temp_predictions is None:
                    temp_predictions = p
                else:
                    temp_predictions = torch.vstack((temp_predictions, p))
        if final_predictions is None:
            final_predictions = temp_predictions
        else:
            final_predictions += temp_predictions

    final_predictions /= n # average the predictions over the number of times that we predicted an image
    sigmoid = nn.Sigmoid()
    for i, output in enumerate(final_predictions):
        probs = sigmoid(output)
        writer.writerow([f'{i}', f'{probs[1]}'])

    avg_metric = metric / count
    return avg_metric


def train(model, criterion, optimizer, num_epochs, train_dataloader, val_dataloader, modelname, eval_metric):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    val_scores = []
    best_val_epoch = -1
    smaller_count = 0
    for epoch in range(num_epochs):
        metric = 0.0
        running_loss = 0.0
        count = 0
        for i_batch, sample_batched in tqdm(enumerate(train_dataloader)):
            inputs, labels = sample_batched['image'].to(device), sample_batched['label'].to(device)

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

        valid_metric = eval_model(model, val_dataloader, eval_metric, criterion=criterion)
        val_scores.append(valid_metric)
        print(
            f"[Epoch {epoch + 1:2d}] Training {eval_metric}: {train_score * 100.0:05.2f}%, Validation accuracy: {valid_metric * 100.0:05.2f}%")

        if len(val_scores) == 1 or valid_metric > val_scores[best_val_epoch]:
            print(
                f'Validation {eval_metric} increased({val_scores[best_val_epoch] * 100.0:05.2f}%--->{valid_metric * 100.0:05.2f}%) \t Saving The Model')
            best_val_epoch = epoch
            smaller_count = 0

            # Saving State Dict
            torch.save(model.state_dict(), f'./models/{modelname}.pth')

        if valid_metric < val_scores[best_val_epoch]:
            smaller_count += 1

        if smaller_count >= 5:
            break

    model.load_state_dict(torch.load(f'./models/{modelname}.pth'))
    best_model = model
    return best_model, val_scores


if __name__ == '__main__':
    # Load the dataloaders from the dataset
    print("Started loading Dataloaders...")
    train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders(
        crop_transform=settings.crop_transform, DA_transform=settings.DA_transform, TTA_transform=settings.TTA_transform)
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