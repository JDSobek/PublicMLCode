import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# setting these to avoid some cupy compilation errors
os.environ["CUDA_PATH"] = "/usr/local/cuda-11.1"
os.environ["LD_LIBRARY_PATH"] = "$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

import cupy as cp
from cupyx.scipy.ndimage import convolve as xconvolve
import numpy as np
import nibabel as nib
from typing import Tuple
import math


def boundingboxextractor(maskarray: np.ndarray, maskvalues=(1)) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], np.ndarray]:
    """Finds the smallest 3-D box that bounds the entire volume of all specified masks.
    Returns:
         the coordinates in the original mask where the bounding box begins (least x, y, and z)
         the dimensions of the bounding box
         a cropped version of the mask"""

    # find z-slices in the mask array containing any mask values of interest, creates 1-D array
    zslices = np.any(np.isin(maskarray, maskvalues), axis=(0, 1))
    # creates a list of indices of slices that contain the masks of interest
    truezslices = np.nonzero(zslices)
    # find x-slices in the mask array containing any mask values of interest, creates 1-D array
    xslices = np.any(np.isin(maskarray, maskvalues), axis=(1, 2))
    # creates a list of indices of slices that contain the masks of interest
    truexslices = np.nonzero(xslices)
    # find y-slices in the mask array containing any mask values of interest, creates 1-D array
    yslices = np.any(np.isin(maskarray, maskvalues), axis=(0, 2))
    # creates a list of indices of slices that contain the masks of interest
    trueyslices = np.nonzero(yslices)

    min_x = np.min(truexslices)
    max_x = np.max(truexslices)
    min_y = np.min(trueyslices)
    max_y = np.max(trueyslices)
    min_z = np.min(truezslices)
    max_z = np.max(truezslices)

    # Calculates overlap as the smallest contiguous region containing every slice in the image array where regions of interest in the mask exist
    boundingbox: np.ndarray = maskarray[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
    boundingbox_firstcorner = (min_x, min_y, min_z)
    boundingbox_dims = (max_x-min_x, max_y-min_y, max_z-min_z)

    return boundingbox_firstcorner, boundingbox_dims, boundingbox


def find_largest_box(input_array: np.ndarray, min_x=1, min_y=1, min_z=1, min_frac=1) -> Tuple[int, int, int]:
    """Find the largest box of 1's in the input array of 0's and 1's
    Input is a mask array and the minimum size of the box you wish to find, as well as the minimum fraction of the box
    that must lie within the ROI.
    Output is largest box's dimensions and the first corner of the box (aka the corner with the min x, min y, and min z)
    """
    # convert to cupy so we can convolve on the gpu
    input_array = cp.array(input_array)

    largest_vol = min_x*min_y*min_z # initialize at minimum volume
    largest_box_dims = (min_x, min_y, min_z) # initialize at minimum dimensions for the box

    test_kernel = cp.ones((min_x, min_y, min_z), dtype=int)
    test_output = cp.zeros(input_array.shape)
    xconvolve(input=input_array, weights=test_kernel, output=test_output, mode='constant')
    if test_output.max() < min_frac*min_x*min_y*min_z:
        print("Minimum size kernel does not fit satisfactorily within ROI.")
    else:
        print("Minimum size kernel fits satisfactorily within ROI.")
    del test_kernel
    del test_output

    # k_valid etc. are to check we're not iterating boxes that are too big in every dimension
    # initialize each length at their minimum every time we iterate its outer loop
    k_valid = True
    k = min_z
    while k <= input_array.shape[2] and k_valid:
        print("k: " + str(k))
        test_k_kernel = cp.ones((min_x, min_y, k), dtype=int)
        test_k_output = cp.zeros(input_array.shape)
        xconvolve(input=input_array, weights=test_k_kernel, output=test_k_output, mode='constant')
        if test_k_output.max() < min_frac * min_x * min_y * k:
            k_valid = False
        else:
            j_valid = True
            j = min_y
            while j <= input_array.shape[1] and j_valid:
                print("j: " + str(j))
                test_j_kernel = cp.ones((min_x, j, k), dtype=int)
                test_j_output = cp.zeros(input_array.shape)
                xconvolve(input=input_array, weights=test_j_kernel, output=test_j_output, mode='constant')
                if test_j_output.max() < min_frac * min_x * j * k:
                    j_valid = False
                else:
                    # optimizing i search pattern
                    i_valid = True
                    min_i = min_x
                    max_i = input_array.shape[0]
                    i = min_i + math.floor((max_i-min_i)/2)
                    while i <= input_array.shape[0] and i_valid:
                        print("i: " + str(i))
                        # initial the convolution kernel to the size of box we're checking
                        outkernel = cp.ones((i, j, k), dtype=int)
                        outconvolve_output = cp.zeros(input_array.shape)
                        # checking whether the kernel fits within our mask
                        xconvolve(input=input_array, weights=outkernel, output=outconvolve_output, mode='constant')

                        inkernel = cp.ones((i-1, j, k), dtype=int)
                        inconvolve_output = cp.zeros(input_array.shape)
                        # checking whether the kernel fits within our mask
                        xconvolve(input=input_array, weights=inkernel, output=inconvolve_output, mode='constant')

                        if outconvolve_output.max() < (min_frac*i*j*k) and inconvolve_output.max() == (min_frac*(i-1)*j*k):
                            print("inner fits, outer does not fit, this is the size")
                            i_valid = False
                            x = i-1 # i is one too large for this combination of j and k so we subtract one
                            y = j
                            z = k

                            # check if this box has a greater volume than the largest box so far
                            if x*y*z > largest_vol:
                                largest_vol = x*y*z
                                largest_box_dims = (x, y, z)
                                print("largest volume: " + str(largest_vol))
                                print("largest box dims: " + str(largest_box_dims))

                        elif outconvolve_output.max() < (min_frac*i*j*k) and inconvolve_output.max() < (min_frac*(i-1)*j*k):
                            print("inner does not fit, outer does not fit, i is too big")
                            # go to halfway between current i and minimum i for this j
                            max_i = i
                            i = min_i + math.floor((i-min_i)/2)
                        elif outconvolve_output.max() == (min_frac*i*j*k) and inconvolve_output.max() == (min_frac*(i-1)*j*k):
                            print("inner fits, outer fits, i is too small")
                            # go to halfway between current i and maximum i for this j
                            min_i = i
                            i = i + math.floor((max_i-i)/2)
                        else:
                            print("this should not be possible")

                    # back in j loop
                    j+=1
            # back in k loop
            k+=1
    # end of k loop
    return largest_box_dims


def largest_box_start_finder(input_array: np.ndarray, largest_box_dims: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Finds the coordinates where the largest box within the input_array starts"""
    input_array = cp.array(input_array)
    kernel = cp.ones(largest_box_dims, dtype=int)
    convolved = cp.zeros(input_array.shape)
    xconvolve(input=input_array, weights=kernel, output=convolved, mode='constant')
    max_val = convolved.max()

    # find z-slices in the mask array containing any mask values of interest, creates 1-D array
    zslices = cp.any(convolved==max_val, axis=(0, 1))
    # find x-slices in the mask array containing any mask values of interest, creates 1-D array
    xslices = cp.any(convolved==max_val, axis=(1, 2))
    # find y-slices in the mask array containing any mask values of interest, creates 1-D array
    yslices = cp.any(convolved==max_val, axis=(0, 2))

    # creates a list of indices of slices that contain the masks of interest
    truezslices = cp.nonzero(zslices)
    # creates a list of indices of slices that contain the masks of interest
    truexslices = cp.nonzero(xslices)
    # creates a list of indices of slices that contain the masks of interest
    trueyslices = cp.nonzero(yslices)

    # find the center coordinate of the first place the maximum size box fits
    box_x = cp.asnumpy(cp.min(truexslices[0]))
    box_y = cp.asnumpy(cp.min(trueyslices[0]))
    box_z = cp.asnumpy(cp.min(truezslices[0]))

    # find the lengths from the center to the edge of the box
    half_box_x = math.floor(largest_box_dims[0]/2)
    half_box_y = math.floor(largest_box_dims[1]/2)
    half_box_z = math.floor(largest_box_dims[2]/2)

    # find that top coordinate
    # because of padding this can be negative and outside of the bounding box
    top_corner_x = box_x - half_box_x
    top_corner_y = box_y - half_box_y
    top_corner_z = box_z - half_box_z

    return (top_corner_x, top_corner_y, top_corner_z)


def box_finder(niftipath: str, min_x=1, min_y=1, min_z=1):
    niftimask = nib.load(niftipath)
    mask_array = np.array(niftimask.dataobj)
    boundingbox_start, boundingbox_size, boundingbox = boundingboxextractor(mask_array)
    large_box_dims = find_largest_box(boundingbox, min_x, min_y, min_z)
    largest_box_start = largest_box_start_finder(boundingbox, large_box_dims)

    # start of the box in image space is the start of the bounding box relative to the image
    # plus the start of the largest box relative to the bounding box
    box_start = [   boundingbox_start[0] + largest_box_start[0],
                    boundingbox_start[1] + largest_box_start[1],
                    boundingbox_start[2] + largest_box_start[2] ]

    return box_start, large_box_dims


if __name__ == '__main__':
    maskpath = "Mask.nii.gz"
    niftipath = "Scan.nii.gz"

    start_coords, lengths = box_finder(maskpath, min_x=96, min_y=96, min_z=4)

    image = nib.load(niftipath)
    image_array = np.array(image.dataobj)
    mask_array = np.array(nib.load(maskpath).dataobj)
    cropped_array = image_array[start_coords[0]:start_coords[0]+lengths[0], start_coords[1]:start_coords[1]+lengths[1], start_coords[2]:start_coords[2]+lengths[2]]
    cropped_mask = mask_array[start_coords[0]:start_coords[0]+lengths[0], start_coords[1]:start_coords[1]+lengths[1], start_coords[2]:start_coords[2]+lengths[2]]

    print("Checks")
    print(lengths)
    print(lengths[0]*lengths[1]*lengths[2])
    print(cp.sum(cp.array(cropped_mask)))

    print("Saving image")
    cropimg = nib.Nifti1Image(cropped_array, image.affine)
    nib.save(cropimg, "test_box.nii.gz")

    cropmask = nib.Nifti1Image(cropped_mask, image.affine)
    nib.save(cropmask, "test_box_mask.nii.gz")
