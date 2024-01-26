import os
import time
import nibabel as nib
import numpy as np
from torch import from_numpy, save, load, stack, permute, zeros, cat
import torchvision
from torch.utils.data import Dataset

DATASET_PATH = 'RawData'


def create_dir_list(path):
    dir_list = []
    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            dir_list.append(d)

    print("Directory list created...")
    return dir_list


def create_ids(dir_list):
    ids = []
    for i in range(len(dir_list)):
        id = dir_list[i][(len(dir_list[i])-3):]
        ids.append(id)

    print("IDs created...")
    return ids


def pre_process(nii):
    # convert from nii -> numpy -> tensor
    t = from_numpy(np.array(nii.dataobj, dtype='float32'))
    # reorder dimensions
    p = permute(t, (2, 1, 0))
    # downsize
    d = torchvision.transforms.Resize(size=(64, 64))(p)
    # to prevent rounding issues when downsampling during training, we need the number of slices to be even
    # so we add an extra slice of just zeros
    zero_padding = zeros(1, 64, 64)
    return cat([d, zero_padding], 0)


def create_dataset():
    start_time = time.time()

    dir_list = create_dir_list(DATASET_PATH)
    ids = create_ids(dir_list)

    # Sample 355 has a malformed file name
    # old_name = f"{DATASET_PATH}/BraTS20_Training_355/W39_1998.09.19_Segm.nii"
    # new_name = f"{DATASET_PATH}/BraTS20_Training_355/BraTS20_Training_355_seg.nii"
    # os.rename(old_name, new_name)

    for id in ids:
        current = f"BraTS20_Training_{id}"

        # open the files for each modality
        nii_flair = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_flair.nii")
        nii_t1 = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_t1.nii")
        nii_t1ce = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_t1ce.nii")
        nii_t2 = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_t2.nii")
        nii_mask = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_seg.nii")

        # convert from nii to tensor
        flair = pre_process(nii_flair)
        t1 = pre_process(nii_t1)
        t1ce = pre_process(nii_t1ce)
        t2 = pre_process(nii_t2)
        mask = pre_process(nii_mask)

        # combine modalities into a single sample
        sample = stack((flair, t1, t1ce, t2), 0)

        # create directory for each id
        os.mkdir(f"TrainingData/{id}")

        # save tensors in new directory
        save(sample, f"TrainingData/{id}/sample.pt")
        save(mask, f"TrainingData/{id}/mask.pt")

    print(f"Dataset created in {time.time()-start_time:.2f} seconds")


class BraTSDataset(Dataset):
    """a class to represent a dataset of multi-modal MR images"""
    def __init__(self, directory, transform=None):
        self.dir = directory
        self.transform = transform

    def __len__(self):
        """A function to retrieve the number of samples in the dataset"""
        length = 0
        for root, dirs, files in os.walk('TrainingData'):
            length += len(dirs)
        return length

    def __getitem__(self, item):
        """Load a given sample from the dataset"""

        # all sample directory names are 3 characters long
        # so concatenate zeros to the start of the directory name where necessary
        prefix = ''
        if item < 10:
            prefix = '00'
        elif item < 100:
            prefix = '0'

        item_dir = f"TrainingData/{prefix}{item}"

        sample = load(item_dir + "/sample.pt")
        mask = load(item_dir + "/mask.pt")

        # apply transformation if one has been given
        if self.transform is not None:
            t = self.transform(sample=sample, mask=mask)
            sample = t["sample"]
            mask = t["mask"]

        return sample, mask

