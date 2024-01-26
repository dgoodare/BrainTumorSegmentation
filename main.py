import torch
from train import TrainingLoop


def get_sample_shape():
    sample = torch.load("TrainingData/001/sample.pt")
    return sample.shape


batch_size = 8
img_size = get_sample_shape()[0]
img_channels = 4  # this is the number of modalities
epochs = 1
learning_rate = 1e-4
model_name = "base"
prune = False

trainer = TrainingLoop(batch_size, img_size, img_channels, epochs, learning_rate, model_name, prune)
trainer.train()

