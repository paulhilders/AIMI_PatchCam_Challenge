import torchvision
from torchvision.transforms import (
    Compose,
    RandomApply,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)

model = 'densenet201'
loss = 'cross_entropy'
optimizer = 'adam'
lr = 0.0001
num_epochs = 8
train = False
freeze = False
modelname = 'ViT_pretrained'
eval_metric = 'auc'
# transform = torchvision.transforms.Compose([
#                                                torchvision.transforms.CenterCrop(32),
#                                            ])
# transform = Compose([
#                         RandomHorizontalFlip(p=0.5),
#                         RandomVerticalFlip(p=0.5),
#                         RandomApply([RandomRotation((90, 90))], p=0.5),
#                     ])