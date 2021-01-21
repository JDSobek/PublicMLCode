"""Converting nifti masks into pytorch cuda tensors saved in .pt format
"""

#import pathlib
import nibabel as nib
import numpy as np
import torch
from typing import Dict, List, Tuple
import os


def train_test_split(keylist: List[str], mod_number = 5) -> Tuple[List[str], List[str]]:
    """Splits a list of dictionary keys into two lists of keys, a testkeys list containing one entry for every mod_number
    entries and a trainkeys list with the rest.  Returns as a tuple (trainkeys, testkeys)"""
    import random
    # shuffle the keys
    random.shuffle(keylist)
    trainkeys = []
    testkeys = []
    #assign keys into each empty list
    for i in range(len(keylist)):
        if i % mod_number == 0:
            testkeys.append(keylist[i])
        else:
            trainkeys.append(keylist[i])
    return trainkeys, testkeys


def data_dict_prep(data_folder: str, mask_label: str) -> Dict[str, List[str]]:
    """Returns a dictionary of paths for nifti images and their associated masks.  Only returns entries for which
        both an image and a mask exist.  Image and mask must have the same name except for the mask_label.
        mask_label must be at the end of the file name (like how rilcontour saves ROI masks)
        Example mask filename format: Patient_0010_Multiple_ROI_Mask.nii.gz
        Returns a dictionary containing keys that are the filenames without the extensions
        and the entries are lists of the form [full image path, full mask path] for each respective file.
        """
    imagedict = {}
    maskdict = {}
    combineddict = {}
    # assign nifti images to imagedict and masks to maskdict
    for dirpath, subdirs, files in os.walk(data_folder):
        for file in files:
            # making sure we're looking at a nifti file, skipping if not
            if file.endswith(mask_label+'.nii.gz') or file.endswith(mask_label+'.nii'):
                niftikey = niftiKeyExtractor(file)
                maskdict[niftikey] = os.path.join(dirpath, file)
            elif file.endswith('.nii.gz') or file.endswith('.nii'):
                # split filename into key and label after removing extension
                niftikey = niftiKeyExtractor(file)
                imagedict[niftikey] = os.path.join(dirpath, file)
            else:
                # print('Non-Nifti file, excluding: ' + file)
                continue
    """
    # assign nifti masks in maskpath to maskdict
    for dirpath, subdirs, files in os.walk(maskpath):
        for file in files:
            # making sure we're looking at a nifti file, skipping if not
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                # split filename into key and label after removing extension
                niftikey = niftiKeyExtractor(file)
                maskdict[niftikey] = os.path.join(dirpath, file)
            else:
                # print('Non-Nifti file, excluding: ' + file)
                continue
    """
    # go through imagedict and assign every entry with a matching mask to combineddict
    for key in imagedict.keys():
        try:
            mask_key = key + mask_label
            combineddict[key] = [imagedict[key], maskdict[mask_key]]
        except KeyError:
            print("Image Key does not have a matching mask, skipping key: " + key)
            continue
    return combineddict


def nibimagemaskdict(imagepath: str, maskpath: str) -> Dict[str,List[str]]:
    """Returns a dictionary of paths for nifti images and their associated masks.  Only returns entries for which
    both an image and a mask exist.  Image and mask must have the same name and exist in separate directories imagepath
    and maskpath respectively.  Returns a dictionary containing with keys that are the filenames without the extensions
    and the entries are lists of the form [full image path, full mask path] for each respective file.
    """
    imagedict = {}
    maskdict = {}
    combineddict = {}
    # assign nifti images in imagepath to imagedict
    for dirpath, subdirs, files in os.walk(imagepath):
        for file in files:
            # making sure we're looking at a nifti file, skipping if not
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                # split filename into key and label after removing extension
                niftikey = niftiKeyExtractor(file)
                imagedict[niftikey] = os.path.join(dirpath, file)
            else:
                #print('Non-Nifti file, excluding: ' + file)
                continue
    # assign nifti masks in maskpath to maskdict
    for dirpath, subdirs, files in os.walk(maskpath):
        for file in files:
            # making sure we're looking at a nifti file, skipping if not
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                # split filename into key and label after removing extension
                niftikey = niftiKeyExtractor(file)
                maskdict[niftikey] = os.path.join(dirpath, file)
            else:
                #print('Non-Nifti file, excluding: ' + file)
                continue
    # go through imagedict and assign every entry with a matching mask to combineddict
    for key in imagedict.keys():
        try:
            mask_key = key + '_Multiple_ROI_Mask'
            combineddict[key] = [imagedict[key], maskdict[mask_key]]
        except KeyError:
            print("Image Key does not have a matching mask, skipping key: " + key)
            continue
    return combineddict


def torchTensorWriter(imagemask_dict: Dict[str, List[str]], norm_values: Dict[str, List[float]], savedir: str, test=False) -> None:
    """Takes dictionaries indexed by filename containing image and mask paths (Re: nibimagemaskdict),
    the volume normalization values for those niftis, and the directory you wish to save the files in.
    Set test=True to save files into the test/validation directory.  Returns nothing.
    Saves the tensors in a structure that looks like:
        images
            test_data
            train_data
        masks
            test_data
            train_data"""

    total_counter = len(imagemask_dict.keys())
    progress_counter = 0
    # iterate over the image+mask dictionary
    for key in imagemask_dict.keys():
        progress_counter += 1
        print("Processing Image " + str(progress_counter) + "/" + str(total_counter))
        # extract paths to mask and image
        imagepath = imagemask_dict[key][0]
        maskpath = imagemask_dict[key][1]

        # copying the arrays from the nifti files is necessary to manipulate the data
        image = np.copy(nib.load(imagepath).dataobj).astype('float32')
        mask = np.copy(nib.load(maskpath).dataobj).astype('float32')
        imagetensor = torch.tensor(image, dtype=torch.float32).cpu()
        masktensor = torch.tensor(mask, dtype=torch.float32).cpu()

        # normalize the image and apply windowing
        # extract volume normalization values for image
        #mean = norm_values[key][0]
        #std_dev = norm_values[key][1]
        #norm_image = (imagetensor-mean)/std_dev
        # since using CT I'm just going to apply windowing and normalize based on that
        # torchvision models expect data to be between [0,1]

        # for aorta try window -30 to 700
        ten24 = torch.tensor(1024.)
        negten24 = torch.tensor(-1024.)
        norm_image = torch.where(imagetensor[:,:,:]>1024., ten24, imagetensor)
        norm_image = torch.where(norm_image[:,:,:]<-1024., negten24, norm_image)
        norm_image = (norm_image+1024.)/2048.

        # iterate over the image slices
        for zslice in range(imagetensor.size()[2]):
            # build the model input that needs to be saved out, for resnet only using 1 slice for each channel
            inputtensor = fullImageInputBuilder(zslice, norm_image).clone().cpu()
            #inputtensor = norm_image[:, :, zslice].clone().cpu()
            # extract the slice mask from the mask tensor and unsqueeze so the shape will match the input tensor
            # should have more channels for multi-mask outputs
            slicemask = masktensor[:,:,zslice].clone().cpu()
            #slicemask = torch.unsqueeze(slicemask, dim=0)

            # saving data out to separate image and mask files with identical names in different folders
            # file names are the image filename + the axis=2 index
            # to filter conditionally add extra if statement to check where the mask exists
            #if torch.any(slicemask):
            # save file
            # saving data to test directory
            if test:
                imagepath = savedir + "images/test_data/" + key + '_' + str(zslice) + '.pt'
                maskpath = savedir + "masks/test_data/" + key + '_' + str(zslice) + '.pt'
                print("Saving file: " + imagepath)
                torch.save(inputtensor, imagepath)
                print("Saving file: " + maskpath)
                torch.save(slicemask, maskpath)
            # saving data to train directory
            else:
                imagepath = savedir + "images/train_data/" + key + '_' + str(zslice) + '.pt'
                maskpath = savedir + "masks/train_data/" + key + '_' + str(zslice) + '.pt'
                print("Saving file: " + imagepath)
                torch.save(inputtensor, imagepath)
                print("Saving file: " + maskpath)
                torch.save(slicemask, maskpath)
    return None


def niftiKeyExtractor(niftipath: str) -> str:
    """Extracts the key for the nifti path string fed into the function.
    Returns the filename without address or the .nii.gz extension."""
    if niftipath.endswith('.nii.gz'):
        niftikey = niftipath.split("/")[-1][:-7]
    if niftipath.endswith('.nii'):
        niftikey = niftipath.split("/")[-1][:-4]
    return niftikey


def niftiPreCalcs(imagemaskdict: Dict[str, List[str]]) -> Tuple[Dict[str, List[float]], List[Tuple[str, int]]]:
    """Does an initial round of the calculations required to preprocess data, in this case calculating volume mean and std_dev
    and returning a dictionary containing these normalization values and a list of tuples containing pairs of slice numbers
    and paths to the image that slice belongs to.  Purpose is to provide a way to avoid repeating lengthy calculations
    for every slice you want to read in and to provide a way to ensure the slices you read are chosen randomly from all
    slices within the dataset.

    Input is a dictionary containing lists of nifti image paths with image data at index 0 and mask data at index 1
    The dictionary returned contains some image identifying tag as keys and tuples of the mean and standard deviation
    for the image in tensor form.
    The list returned contains every image and slice pair in the dataset, with the image stored as the key value from
    the input dictionary and the slice stored by its axis=2 index."""
    # initializing the data structures that will be returned
    value_dict = {}
    slice_indexes = []
    # looping over every element of the input dataset
    keylist = list(imagemaskdict.keys())
    for key in keylist:
        print("Precalculating parameters for image: " + key)
        # python prefers you initialize your dictionary of lists with an empty list
        value_dict[key] = []
        # convert the nifti image into a torch tensor for normalization calculations
        niftiarray = np.copy(nib.load(imagemaskdict[key][0]).dataobj)
        niftitensor = torch.tensor(niftiarray, dtype=torch.float32).cpu()#.cuda()
        # calculate values for normalization
        mean = torch.mean(niftitensor)
        std_dev = torch.std(niftitensor)
        # insert the normalization values into the empty list created at this dictionary key
        value_dict[key] = [mean, std_dev]
        # assemble the list of all slices with the paths to the image they originate from
        for zslice in range(niftiarray.shape[2]):
            slice_indexes.append((key, zslice))
    return value_dict, slice_indexes


def fullImageInputBuilder(sliceindex: int, norm_image: torch.tensor) -> torch.tensor:
    """Creates the input tensors.
    Input a normalized tensor (preferably from the whole scan) and the slice index of that tensor the input tensor
    is to be centered around.
    Returns the 5-slice array corresponding to the slice and image pair given as input."""

    # Torch requires channel first format such that data format is (batchsize, channel, height, width)
    # assemble the 5 slice tensors with special cases for the top and bottom of the scan
    # reshaping is necessary for using torch.cat to combine single slices into higher dimensional arrays
    if sliceindex == 0:
        #inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex], (512,512,1)), torch.reshape(norm_image[:, :, sliceindex], (512,512,1)), norm_image[:, :, sliceindex:sliceindex + 3]), dim=2)
        inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 2], (1, 512, 512))), dim=0).cpu()#.cuda()
    elif sliceindex == 1:
        #inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex - 1], (512,512,1)), norm_image[:, :, sliceindex - 1:sliceindex + 3]), dim=2)
        inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex - 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex - 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 2], (1, 512, 512))), dim=0).cpu()#.cuda()
    elif sliceindex == norm_image.size()[2] - 2:
        #inputslices = torch.cat((norm_image[:, :, sliceindex - 2: sliceindex + 1], torch.reshape(norm_image[:, :, sliceindex + 1], (512,512,1)), torch.reshape(norm_image[:, :, sliceindex + 1], (512,512,1))), dim=2)
        inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex - 2], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex - 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 1], (1, 512, 512))), dim=0).cpu()#.cuda()
    elif sliceindex == norm_image.size()[2] - 1:
        #inputslices = torch.cat((norm_image[:, :, sliceindex - 2:sliceindex], torch.reshape(norm_image[:, :, sliceindex], (512,512,1)), torch.reshape(norm_image[:, :, sliceindex], (512,512,1)), torch.reshape(norm_image[:, :, sliceindex], (512,512,1))), dim=2)
        inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex - 2], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex - 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512))), dim=0).cpu()#.cuda()
    else:
        #inputslices = norm_image[:, :, sliceindex - 2:sliceindex + 3]
        inputslices = torch.cat((torch.reshape(norm_image[:, :, sliceindex - 2], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex - 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 0], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 1], (1, 512, 512)),
                                 torch.reshape(norm_image[:, :, sliceindex + 2], (1, 512, 512))), dim=0).cpu()#.cuda()
    return inputslices


if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # images and masks are buried deep in subdirectories thanks to how dicoms do things, may need a more complicated way
    # to deal with extracting their paths
    # rework nibimagemaskdict to take the overall path and sort masks from images
    image_mask_dir = "/nifti_masks/dicoms"

    data_dict = data_dict_prep(image_mask_dir, "_Multiple_ROI_Mask")

    # for windowing normalization I won't use these statistics so skipping this calculation
    #niftinormvalues, image_slice_indices = niftiPreCalcs(data_dict)
    niftinormvalues = {}

    key_list = list(data_dict.keys())

    train_data_keys, test_data_keys = train_test_split(key_list)

    # save directories for new data
    datadir = "/Data/test_tensors/"

    # splitting data into train and test sets
    train_data_dict = {}
    test_data_dict = {}

    for key in list(train_data_keys):
        train_data_dict[key] = data_dict[key]

    for key in list(test_data_keys):
        test_data_dict[key] = data_dict[key]


    # save test set
    torchTensorWriter(test_data_dict, niftinormvalues, datadir, test=True)

    # save train set
    torchTensorWriter(train_data_dict, niftinormvalues, datadir, test=False)
