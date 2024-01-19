import torch
from train import TrainingLoop


def get_sample_shape():
    sample = torch.load("TrainingData/001/sample.pt")
    return sample.shape


batch_size = 8
img_size = get_sample_shape()[0]
modalities = 4
epochs = 1
learning_rate = 1e-4

trainer = TrainingLoop(batch_size, img_size, modalities, epochs, learning_rate)
trainer.train()

