import torch
import torch.nn as nn

from Dataloader import getDataLoaders
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import h5py
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")

def getDataSets():
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_train_x.h5', 'r')
    trainset_x = f['x']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_train_y.h5', 'r')
    trainset_y = f['y']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_test_x.h5', 'r')
    testset_x = f['x']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_test_y.h5', 'r')
    testset_y = f['y']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_valid_x.h5', 'r')
    validset_x = f['x']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_valid_y.h5', 'r')
    validset_y = f['y']
    train_set = [{"x": torch.from_numpy(trainset_x[i]).permute(2, 0, 1).float(), "labels": torch.from_numpy(trainset_y[i]).squeeze()} for i in range(10)]
    test_set = [{"x": torch.from_numpy(testset_x[i]).permute(2, 0, 1).float(), "labels": testset_y[i]} for i in range(len(testset_y))]
    valid_set = [{"x": torch.from_numpy(validset_x[i]).permute(2, 0, 1).float(), "labels": torch.from_numpy(validset_y[i]).squeeze()} for i in range(10)]
    return train_set, test_set, valid_set

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    train_set, test_set, valid_set = getDataSets()
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    metric = evaluate.load("accuracy")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
    )
    trainer.train()
