import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from datetime import datetime

from dataset import BraTSDataset
from loss import DiceLoss
from model import UNet
from visualiser import Visualiser


class TrainingLoop:
    """A class that encapsulates the creation and training of a UNet segmentation model"""
    def __init__(self, batch, size, channels, epochs, lr):
        """
        Initialise model, dataset, loss function, and any hyperparameters
        that will be used during the training loop
        """
        self.batch_size = batch
        self.img_size = size
        self.img_channels = channels
        self.epochs = epochs
        self.learning_rate = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataloader = self.create_dataloader()  # the 'dataset'
        self.loss = DiceLoss()  # loss function
        self.model = UNet(in_channels=self.img_channels).to(self.device)  # the UNet model that is being trained
        self.optimiser = optim.Adam(params=self.model.parameters(),
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999),  # these are the amount that the LR will decay by over time
                                    eps=1e-08  # this is a very small value to prevent a division by zero
                                    )
        self.scaler = torch.cuda.amp.GradScaler()
        self.visualiser = Visualiser("logs", True)  # for data visualisation stuff

    def create_dataloader(self):
        """Initialise the dataloader"""
        # define transformations
        t = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0]
                ),
                transforms.ToTensor(),
            ]
        )
        dataset = BraTSDataset('TrainingData')  # TODO: add transforms later
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    @staticmethod
    def get_input_slices(input, slice_no):
        """Extracts a set of example slices for each modality from the current batch"""
        flair = input[0][slice_no]
        t1 = input[0][slice_no]
        t1ce = input[0][slice_no]
        t2 = input[0][slice_no]

        return [flair, t1, t1ce, t2]

    @staticmethod
    def get_mask_slice(mask, slice_no):
        """Get an example slice from a mask (this can be either a prediction or a target)"""
        return mask[slice_no]

    def save_model(self, filename):
        """Save an iteration of a (semi)trained model"""
        model = {
            'Model': self.model,
            'Model state': self.model.state_dict(),
            'Optimiser': self.optimiser.state_dict()
        }
        torch.save(model, filename)
        print(f"{filename} saved")

    def print_model_details(self):
        print(f"Batch size : {self.batch_size} \n"
              f"Image channels : {self.img_channels} \n"
              f"Epochs : {self.epochs} \n"
              f"Learning rate : {self.learning_rate} \n"
              f"Device : {self.device} \n"
              )

    def train(self):
        """The main training loop"""
        self.print_model_details()

        for epoch in range(self.epochs):
            print(
                "\n==============================================\n"
                f"Epoch [{epoch + 1}/{self.epochs}] \n"
                "==============================================\n"
            )

            accumulation_steps = 5

            for batch_idx, sample in enumerate(self.dataloader):
                print(f"batch {batch_idx}...")
                # TODO: there will likely be some data manipulation needing to be done here
                # extract the 4 modalities from the sample
                input = sample[0].to(self.device)
                # extract the target mask
                target = sample[1].to(self.device)

                # forward pass
                print("starting forward pass...")
                with torch.cuda.amp.autocast():
                    prediction = self.model(input)
                    loss = self.loss.forward(prediction, target)

                # backward pass
                print("starting backward pass...")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

                if (batch_idx+1) % accumulation_steps == 0:
                    self.optimiser.step()
                    self.optimiser.zero_grad()

                squeezed = torch.squeeze(prediction, 1)

                # update logs every 10 batches
                if batch_idx % 10 == 0:
                    print(f"updating logs...")
                    print(f"batch {batch_idx}/{len(self.dataloader)}")

                    # get some examples slices from the first sample in the current batch
                    slice_no = 50
                    input_slices = self.get_input_slices(input[0], slice_no)

                    prediction_slice = self.get_mask_slice(squeezed[0], slice_no)
                    target_slice = self.get_mask_slice(target[0], slice_no)

                    # visualise the extracted slices and the current loss value
                    self.visualiser.plot(input_slices, prediction_slice, target_slice, loss)
                    self.visualiser.step += 1

        # save the trained model
        filename = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".pth"
        self.save_model(filename)

