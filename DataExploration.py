import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
import skimage.transform as trans
from skimage.transform import rotate
from skimage.transform import resize
import shutil

# neural imaging libraries
import nibabel as nib
import gif_your_nifti.core as nif2gif # handy library for displaying multi-slice data as a gif

# pytorch libraries

DATASET_PATH = 'RawData/'

def get_data_shape():
    test_flair = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
    test_t1 = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii')
    test_t1ce = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii')
    test_t2 = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii')
    test_mask = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

    print(f"flair shape; {test_flair.shape}")
    print(f"t1 shape; {test_t1.shape}")
    print(f"t1ce shape; {test_t1ce.shape}")
    print(f"flair shape; {test_flair.shape}")
    print(f"flair shape; {test_flair.shape}")


def display_slice(slice):
    # load some example data
    test_flair = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
    test_t1 = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
    test_t1ce = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
    test_t2 = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
    test_mask = nib.nifti1.load(DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()

    # create plot to display a slice from each modality
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
    ax1.imshow(test_flair[:, :, test_flair.shape[0] // 2 - slice], cmap='gray')
    ax1.set_title('flair')
    ax2.imshow(test_t1[:, :, test_t1.shape[0] // 2 - slice], cmap='gray')
    ax2.set_title('t1')
    ax3.imshow(test_t1ce[:, :, test_t1ce.shape[0] // 2 - slice], cmap='gray')
    ax3.set_title('t1ce')
    ax4.imshow(test_t2[:, :, test_t2.shape[0] // 2 - slice], cmap='gray')
    ax4.set_title('t2')
    ax5.imshow(test_mask[:, :, test_mask.shape[0] // 2 - slice], cmap='gray')
    ax5.set_title('mask')
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    ax1.imshow(rotate(montage(test_t1[50:-50, :, :]), 90, resize=True), cmap='gray')
    plt.show()


display_slice(25)
