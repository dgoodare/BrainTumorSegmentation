import os
import time
import nibabel as nib
import numpy as np
from torch import from_numpy, save, load
from torch.utils.data import Dataset

DATASET_PATH = 'RawData'


class BraTSDataset(Dataset):
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
        item_dir = f"TrainingData/{item}"

        flair = load(item_dir + "/flair.pt")
        t1 = load(item_dir + "/t1.pt")
        t1ce = load(item_dir + "/t1ce.pt")
        t2 = load(item_dir + "/t2.pt")
        mask = load(item_dir + "/mask.pt")

        # check if a transformation has been specified
        if self.transform is not None:
            t = self.transform(flair=flair, t1=t1, t1ce=t1ce, t2=t2, mask=mask)
            flair = t["flair"]
            t1 = t["t1"]
            t1ce = t["t1ce"]
            t2 = t["t2"]
            mask = t["mask"]

        return flair, t1, t1ce, t2, mask


def create_dir_list(path):
    dir_list = []
    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            dir_list.append(d)

    print(dir_list)
    # remove entry 355 since it has dodgy filenames
    dir_list.remove(DATASET_PATH + '\\BraTS20_Training_355')

    return dir_list


def create_ids(dir_list):
    ids = []
    for i in range(len(dir_list)):
        id = dir_list[i][(len(dir_list[i])-3):]
        ids.append(id)

    return ids


def create_dataset():
    start_time = time.time()

    dir_list = create_dir_list(DATASET_PATH)
    ids = create_ids(dir_list)

    for id in ids:
        current = f"BraTS20_Training_{id}"

        # open the files for each modality
        nii_flair = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_flair.nii")
        nii_t1 = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_t1.nii")
        nii_t1ce = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_t1ce.nii")
        nii_t2 = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_t2.nii")
        nii_mask = nib.nifti1.load(f"{DATASET_PATH}/{current}/{current}_seg.nii")

        # convert from nii to tensor
        flair = from_numpy(np.array(nii_flair.dataobj, dtype='int16'))
        t1 = from_numpy(np.array(nii_t1.dataobj, dtype='int16'))
        t1ce = from_numpy(np.array(nii_t1ce.dataobj, dtype='int16'))
        t2 = from_numpy(np.array(nii_t2.dataobj, dtype='int16'))
        mask = from_numpy(np.array(nii_mask.dataobj, dtype='int16'))

        # create directory for each id
        os.mkdir(f"TrainingData/{id}")
        # save tensors in new directory
        save(flair, f"TrainingData/{id}/flair.pt")
        save(t1, f"TrainingData/{id}/t1.pt")
        save(t1ce, f"TrainingData/{id}/t1ce.pt")
        save(t2, f"TrainingData/{id}/t2.pt")
        save(mask, f"TrainingData/{id}/mask.pt")

    print(f"Dataset created in {time.time()-start_time:.2f} seconds")

