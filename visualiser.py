import torch
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from shutil import rmtree


class Visualiser:
    def __init__(self, logdir, clean_old=False):
        self.logdir = logdir

        # check if old log data should be removed
        if clean_old:
            self.clean_old_logs()

        self.sample_writer = SummaryWriter(self.logdir + "/sample")
        self.prediction_writer = SummaryWriter(self.logdir + "/prediction")
        self.target_writer = SummaryWriter(self.logdir + "/target")
        self.loss_writer = SummaryWriter(self.logdir + "/loss")

        self.step = 0  # global step value for recording log data

    def clean_old_logs(self):
        """Remove old log data"""
        for path in Path(self.logdir).glob("**/*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)

    def plot(self, samples, predictions, targets, loss):
        # create grids to display slices
        sample_grid = torchvision.utils.make_grid(samples)
        prediction_grid = torchvision.utils.make_grid(predictions)
        target_grid = torchvision.utils.make_grid(targets)

        self.sample_writer.add_image("Input Data", sample_grid, global_step=self.step)
        self.prediction_writer.add_image("Prediction", prediction_grid, global_step=self.step)
        self.target_writer.add_image("Target", target_grid, global_step=self.step)

        self.loss_writer.add_scalar("Loss", loss, global_step=self.step)
