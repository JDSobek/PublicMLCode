import os
import random
import torch
import torchvision
import PIL


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


class MedDataset(torch.utils.data.Dataset):
    # define the dataset used for the data
    def __init__(self, tensor_dir, test=False):
        self.tensor_dir = tensor_dir
        self.image_dict = {}
        self.mask_dict = {}
        # tuple or list of sequential integers 0 to n
        self.mask_labels = (0, 1, 2)

        # set the directories for the data
        if test:
            imagepath = self.tensor_dir + "images/test_data/"
            maskpath = self.tensor_dir + "masks/test_data/"
        else:
            imagepath = self.tensor_dir + "images/train_data/"
            maskpath = self.tensor_dir + "masks/train_data/"
        # fill the dictionaries that store the data locations
        for dirpath, subdirs, files in os.walk(imagepath):
            for file in files:
                key = file
                self.image_dict[key] = os.path.join(dirpath, file)
        for dirpath, subdirs, files in os.walk(maskpath):
            for file in files:
                key = file
                self.mask_dict[key] = os.path.join(dirpath, file)
        self.keys = list(self.image_dict.keys())

    def __getitem__(self, index):
        # get an item (a tensor and its mask) from the dataset
        item_key = self.keys[index]
        # it is recommended not to send the data to the GPU in this function as it incurs extra overhead
        # compared to sending the entire batch to the GPU at once
        # image_tensor = torch.unsqueeze(torch.load(self.image_dict[item_key]), dim=0)#.cuda()
        image_tensor = torch.load(self.image_dict[item_key])
        # adding depth dimension to 2-d mask
        mask_tensor = torch.unsqueeze(torch.load(self.mask_dict[item_key]), dim=0)#.cuda()
        return image_tensor, mask_tensor
        # input_tensor = torch.cat((image_tensor, image_tensor, image_tensor), dim=0)#.cuda()
        # target_tensor = torch.zeros((len(self.mask_labels), mask_tensor.size()[0], mask_tensor.size()[1], mask_tensor.size()[2]))#.cuda()
        # for label in self.mask_labels:
        #     target_tensor[label, :, :, :] = torch.where(mask_tensor == label, 1, 0)
        # return input_tensor, target_tensor

    def __len__(self):
        # return the length of the dataset
        return len(self.keys)
