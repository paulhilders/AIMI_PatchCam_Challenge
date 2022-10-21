import torchvision.transforms as T
import torch

model = 'densenet201'
loss = 'cross_entropy'
optimizer = 'adam'
lr = 0.0001
num_epochs = 5
train = False
freeze = False
modelname = 'densenet201_20epochs(from60)'
eval_metric = 'accuracy'
TTA = True
crop_transform = None  # T.Compose([T.CenterCrop(32),])
DA_transform = None
TTA_transform = T.RandomApply(transforms=torch.nn.ModuleList([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(degrees=90),
                ]), p=0.8)
