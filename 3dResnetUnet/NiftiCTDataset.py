import os
import random
import torch
import torchvision
import PIL
import nibabel as nib
import numpy as np


@torch.jit.script
def adjust_contrast(img: torch.Tensor, ratio: float) -> torch.Tensor:
    """See torchvision functional_tensor adjust_contrast and _blend methods.
    Note tensors should already be saved in range -1 to 1 so this clamps to that range"""
    mean = torch.mean(img)
    return (ratio * img + (1.0 - ratio) * mean).clamp(-1., 1.).to(img.dtype)


def augment_fn(x_tensor, y_tensor):
    """Randomly applies augmentations to the data."""
    # setting probabilities for different augmentations
    if x_tensor.size()[0] != y_tensor.size()[0]:
        raise AttributeError("x and y tensors have different batch sizes")

    aug_prob = 0.9
    contrast_prob = 0.5
    brightness_prob = 0.5
    rot_prob = 0.8
    max_rot = 10. # degrees

    for batch_index in range(x_tensor.size()[0]):
        # setting parameters for degree of augmentation
        rot_angle = random.uniform(0., max_rot)
        delta_contrast = random.uniform(0.95, 1.) # contrast factors over 1 introduce artifacts
        delta_brightness = random.uniform(0.95, 1.05)
        # initializing operations on the y_yensor to False so they are only performed
        # if the operation is performed on the x_tensor
        rotate_mask = False

        bool_augment = random.random() < aug_prob
        if bool_augment and random.random() < contrast_prob:
            # torch's inbuilt contrast adjustment wants 3 channel images, need to use a custom version of it
            x_tensor[batch_index,...] = adjust_contrast(img=x_tensor[batch_index,...], ratio=delta_contrast)
        if bool_augment and random.random() < brightness_prob:
            x_tensor[batch_index,...] = torchvision.transforms.functional.adjust_brightness(img=x_tensor[batch_index,...], brightness_factor=delta_brightness)
        if bool_augment and random.random() < rot_prob:
            x_tensor[batch_index,...] = torchvision.transforms.functional.rotate(img=x_tensor[batch_index,...], angle=rot_angle, resample=PIL.Image.BILINEAR)
            rotate_mask = True

        if rotate_mask:
            y_tensor[batch_index,...] = torchvision.transforms.functional.rotate(img=y_tensor[batch_index,...], angle=rot_angle, resample=PIL.Image.NEAREST)

    return x_tensor, y_tensor


def input_prep(image_tensor, mask_tensor, mask_labels):
    """A bunch of operations that will probably be faster with cuda tensors so I do them here instead of in the dataset"""
    if image_tensor.size()[0] != mask_tensor.size()[0]:
        raise AttributeError("x and y tensors have different batch sizes")

    # adding channel dimensions
    image_tensor = torch.unsqueeze(image_tensor, dim=1)
    mask_tensor = torch.unsqueeze(mask_tensor, dim=1)
    # reshaping inputs and targets
    input_tensor = torch.cat((image_tensor, image_tensor, image_tensor), dim=1)
    target_tensor = torch.zeros((mask_tensor.size()[0], len(mask_labels), mask_tensor.size()[2], mask_tensor.size()[3], mask_tensor.size()[4])).cuda()
    # one-hot encoding masks
    for batch_index in range(image_tensor.size()[0]):
        for label in mask_labels:
            target_tensor[batch_index, label, :, :, :] = torch.where(mask_tensor[batch_index, ...] == label, 1, 0)
    return input_tensor, target_tensor


class NiftiCTDataset(torch.utils.data.Dataset):
    """
    Assumes images and masks are store in same directory, with identical names except for the mask_tag appended at
    the end of the filename.  Directory should be clear of all other nifti files.  mask_tag must be an exact match.
    Does not handle train/test splitting, store files in separate directories for that.
    input_depth is how many axial slices each input to your model uses.
    window_max and window_min determine the highest and lowest values respectively of the window that will be applied to your data.
    mask_labels are the values present in your mask.  These should be sequential integers as they will be used to index the mask array.
    """
    # define the dataset used for the data
    def __init__(self, nifti_dir, window_max=1024., window_min=-1024., mask_labels=(0, 1), mask_tag='_ROI_Mask'):
        # keys will be the image paths, stored values will be the corresponding mask paths
        self.nifti_dict = {}
        # this dictionary is to help assemble the items returned by the Dataset
        self.item_list = []
        # tuple or list of sequential integers 0 to n
        self.mask_labels = mask_labels
        self.mask_label = mask_tag

        self.window_max = window_max
        self.window_min = window_min
        self.window_max_tensor = torch.tensor(self.window_max)
        self.window_min_tensor = torch.tensor(self.window_min)

        # create a dictionary of all the nifti files in the directory
        for dirpath, _, files in os.walk(nifti_dir):
            for file in files:
                image_path = os.path.join(dirpath, file)
                if image_path.endswith(mask_tag + '.nii.gz') or image_path.endswith(mask_tag + '.nii'):
                    # print('File is mask, adding later')
                    continue
                elif image_path.endswith('.nii.gz'):
                    mask_path = image_path[:-7] + mask_tag + '.nii.gz'
                elif image_path.endswith('.nii'):
                    mask_path = image_path[:-4] + mask_tag + '.nii'
                else:
                    # print('Non-Nifti file, skipping')
                    continue
                self.nifti_dict[image_path] = mask_path

        # make sure every entry in the dictionary is accessible and delete any entry that is not
        keylist = list(self.nifti_dict.keys())
        for key in keylist:
            try:
                nib.load(key)
            except nib.filebasedimages.ImageFileError or FileNotFoundError:
                print("File does not exist")
                print(key)
                del self.nifti_dict[key]
                continue
            try:
                nib.load(self.nifti_dict[key])
            except nib.filebasedimages.ImageFileError or FileNotFoundError:
                print("Mask does not exist")
                print(self.nifti_dict[key])
                del self.nifti_dict[key]
                continue

        # put together a dictionary of files to read and slices
        keylist = list(self.nifti_dict.keys()) # get the cleaned key list
        for key in keylist:
            nifti = nib.load(key)
            num_axial_slices = nifti.header.get_data_shape()[2]
            for index in range(num_axial_slices):
                self.item_list.append((key, index))


    def __getitem__(self, index):
        # get an item (an image and a slice index from that image) from the dataset
        item_key = self.item_list[index][0]
        axial_slice = self.item_list[index][1]
        # it is recommended not to send the data to the GPU in the __getitem__ function
        # as it incurs extra overhead compared to sending the entire batch to the GPU at once

        # load the mask and create the target tensor
        mask = nib.load(self.nifti_dict[item_key])
        target = np.copy(mask.slicer[:,:, axial_slice:axial_slice+1].dataobj).astype('float32')
        masktensor = torch.tensor(target, dtype=torch.float32).cpu()
        targettensor = torch.unsqueeze(masktensor[:,:,0], dim=0)

        # load the image and create the input tensor
        image = nib.load(item_key)
        total_slices = image.header.get_data_shape()[2]
        if axial_slice == 0:
            image_slices = np.copy(image.slicer[:, :, axial_slice:axial_slice+3].dataobj).astype('float32')
            imagetensor = torch.tensor(image_slices, dtype=torch.float32).cpu()
            inputtensor = torch.cat((   torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 1], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0)), dim=0)

        elif axial_slice == 1:
            image_slices = np.copy(image.slicer[:, :, axial_slice-1:axial_slice+3].dataobj).astype('float32')
            imagetensor = torch.tensor(image_slices, dtype=torch.float32).cpu()
            inputtensor = torch.cat((   torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 1], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 3], dim=0)), dim=0)

        elif axial_slice == total_slices - 2:
            image_slices = np.copy(image.slicer[:, :, axial_slice-2:axial_slice+2].dataobj).astype('float32')
            imagetensor = torch.tensor(image_slices, dtype=torch.float32).cpu()
            inputtensor = torch.cat((   torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 1], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 3], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 3], dim=0)), dim=0)

        elif axial_slice == total_slices - 1:
            image_slices = np.copy(image.slicer[:, :, axial_slice - 2:axial_slice+1].dataobj).astype('float32')
            imagetensor = torch.tensor(image_slices, dtype=torch.float32).cpu()
            inputtensor = torch.cat((   torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 1], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0)), dim=0)

        else:
            image_slices = np.copy(image.slicer[:, :, axial_slice-2:axial_slice+3].dataobj).astype('float32')
            imagetensor = torch.tensor(image_slices, dtype=torch.float32).cpu()
            inputtensor = torch.cat((   torch.unsqueeze(imagetensor[:, :, 0], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 1], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 2], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 3], dim=0),
                                        torch.unsqueeze(imagetensor[:, :, 4], dim=0)), dim=0)

        # window tensor values and rescale 0 to 1
        # pushing this to input_prep after arrays are pushed to cuda will probably speed things up
        windowed_tensor = torch.where(inputtensor[:,:,:] > self.window_max, self.window_max_tensor, inputtensor)
        windowed_tensor = torch.where(windowed_tensor[:,:,:] < self.window_min, self.window_min_tensor, windowed_tensor)
        windowed_tensor = (windowed_tensor-self.window_min)/(self.window_max - self.window_min)

        return windowed_tensor, targettensor


    def __len__(self):
        # return the length of the dataset
        return len(self.item_list)


if __name__ == '__main__':
    """
    Split the data into training and validation and/or test folders, we need separate datasets for training, validation, and testing.
    Assumes masks and images are stored in identical locations with mask using image's name + the mask label
    """
    pass
