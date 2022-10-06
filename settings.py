import torchvision

model = 'densenet201'
loss = 'cross_entropy'
optimizer = 'adam'
lr = 0.0001
num_epochs = 5
train = False
freeze = False
modelname = 'ViT_pretrained'
eval_metric = 'auc'
transform = torchvision.transforms.Compose([
                                               torchvision.transforms.CenterCrop(32),
                                           ])