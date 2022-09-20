import torch
import torch.optim as optim
import torch.nn as nn
import Dataloader
from tqdm import tqdm

from densenet_pytorch import DenseNet 

# # Open image
# input_image = Image.open("img.jpg")
print("started loading dataloaders")
train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders()
print("dataloaders loaded")
# # Preprocess image
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# # Load class names
# labels_map = json.load(open("labels_map.txt"))
# labels_map = [labels_map[str(i)] for i in range(1000)]

# # Classify with DenseNet121
# model = DenseNet.from_pretrained("densenet121")
# model.eval()

model = DenseNet.from_pretrained("densenet121")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# move the input and model to GPU for speed if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(100):
    train_loss = 0.0
    for data in tqdm(train_dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('train_loss: ', train_loss)
    
    valid_loss = 0.0
    model.eval()
    for data in tqdm(valid_dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        print('valid_loss: ', valid_loss)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f\
        }--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        torch.save(model.state_dict(), './densenet_finetuned.pth')


# with torch.no_grad():
#     logits = model(input_batch)
# preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

# print("-----")
# for idx in preds:
#     label = labels_map[idx]
#     prob = torch.softmax(logits, dim=1)[0, idx].item()
#     print(f"{label:<75} ({prob * 100:.2f}%)")