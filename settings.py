import torchvision

model = 'densenet201'
loss = 'cross_entropy'
optimizer = 'adam'
lr = 0.0001
num_epochs = 50
train = True
freeze = False
modelname = 'densenet201'
eval_metric = 'accuracy'
transform = torchvision.transforms.Compose([
                                               torchvision.transforms.CenterCrop(32)
                                           ])
# freeze = True
# modelname = 'densenet201_freeze'