import numpy as np
import torch
import nibabel as nib
from typing import List
from ResNet3DUNet import ResNet3DUNet
from TorchResnet3dconvs import resnet34


def fullImageInputBuilder(sliceindex: int, norm_image: torch.tensor) -> torch.tensor:
    """Creates the input tensors for 2.5D model.
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


def maskSaver(nifti_paths: List[str], masktopdir: str, torch_model):
    """Save masks generated using a torch model."""
    if next(torch_model.parameters()).device == 'cpu':
        on_cpu = True
    else:
        on_cpu = False

    for nifti_path in nifti_paths:
        print(nifti_path)
        maskfilename = masktopdir + nifti_path[:-7].split("/")[-1] + "_3dResnetUNet_mask.nii.gz"
        image = nib.load(nifti_path)
        if on_cpu:
            image_array = torch.tensor(np.array(image.dataobj), dtype=torch.float32).cpu()
            mask_array = torch.tensor(np.zeros(np.shape(image_array))).cpu()
            ten24 = torch.tensor(1024.).cpu()
            negten24 = torch.tensor(-1024.).cpu()
        else:
            image_array = torch.tensor(np.array(image.dataobj), dtype=torch.float32).cuda()
            mask_array = torch.tensor(np.zeros(np.shape(image_array))).cuda()
            ten24 = torch.tensor(1024.).cuda()
            negten24 = torch.tensor(-1024.).cuda()

        image_array = torch.where(image_array[...] > 1024., ten24, image_array)
        image_array = torch.where(image_array[...] < -1024., negten24, image_array)
        image_array = (image_array + 1024.) / 2048.

        axial_slices = image_array.size()[2]
        for z_slice in range(axial_slices):
            print(str(z_slice) + "/" + str(axial_slices-1))
            # generate mask for slice
            model_input = fullImageInputBuilder(z_slice, image_array)
            # adding the channel dimension
            model_input = torch.unsqueeze(model_input, dim=0).cuda()
            model_input = torch.cat((model_input, model_input, model_input), dim=0)
            # adding the batch dimension
            model_input = torch.unsqueeze(model_input, dim=0)
            # writing the slice prediction to the mask
            slice_mask = torch.argmax(torch_model(model_input), dim=1)
            mask_array[:, :, z_slice] = slice_mask[0, 0, ...]

        mask_nifti = nib.Nifti1Image(mask_array.cpu().numpy(), image.affine)
        nib.save(mask_nifti, maskfilename)
    return None


if __name__ == '__main__':
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    nifti_paths = ["Image.nii.gz"]
    mask_save_path = "/test_segs/"
    modelfilesavepath = r"/Models/3DResnetUNet.pth"

    # seg_model = ResUNet().cuda()
    resnetmodel = resnet34(pretrained=True, is_encoder=True)
    unetmodel = ResNet3DUNet(resnetmodel, input_depth=5, num_classes=2).cuda()
    seg_model = unetmodel
    seg_model.load_state_dict(torch.load(modelfilesavepath))
    seg_model.eval()

    maskSaver(nifti_paths, mask_save_path, seg_model)
