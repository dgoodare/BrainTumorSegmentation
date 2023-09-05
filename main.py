import torch
from train import Trainer


def get_sample_shape():
    sample = torch.load("TrainingData/001/sample.pt")
    return sample.shape


batch_size = 16
img_size = get_sample_shape()[0]
img_channels = 4  # TODO: is this correct?
epochs = 1
learning_rate = 1e-4

trainer = Trainer(batch_size, img_size, img_channels, epochs, learning_rate)
trainer.train()

