"""
Semi-generic script to convert NIfTI segmentations into MedYOLO txt labels.
Only generates entries for specified classes to filter out undesired classes from multi-class segmentations.
Also includes rotation to generate augmented examples.  Best used with high-quality segmentations.

This script makes some assumptions on how your imaging data is stored (see trainvalsplit function).
Rotation assumes CT data and needs a fill value changed if you intend to use it with MR data.
train-val split is 80/20 by default.
"""

import numpy as np
from typing import List
import nibabel as nib
import os
import random
import shutil


def create_label_3D(z, x, y, d, w, h, label_path, class_num = 0):
    """
    Adds class label information to the txt label.
    """
    if os.path.isfile(label_path): 
        new_line = '\n'
    else: 
        new_line = ''
    f = open(label_path, 'a+')
    f.write(f'{new_line}{class_num} {z} {x} {y} {d} {w} {h}')
    f.close()


def findBBox(mask_array: np.ndarray, label_list: List[int]):
    """
    Converts the mask segmentation data into a list of MedYOLO-formatted labels.
    This does not look for disconnected parts or do any clean up of stray pixels.
    """
    bbox_list = []
    
    x_length = mask_array.shape[0]
    y_length = mask_array.shape[1]
    z_length = mask_array.shape[2]
    
    for mask_value in label_list:
        if np.any(np.isin(mask_array, [mask_value])):
            xslices = np.any(np.isin(mask_array, [mask_value]), axis=(1,2))
            yslices = np.any(np.isin(mask_array, [mask_value]), axis=(0,2))
            zslices = np.any(np.isin(mask_array, [mask_value]), axis=(0,1))
        
            truexslices = np.nonzero(xslices)
            trueyslices = np.nonzero(yslices)
            truezslices = np.nonzero(zslices)
            
            min_x = int(truexslices[0].min())
            max_x = int(truexslices[0].max())
            min_y = int(trueyslices[0].min())
            max_y = int(trueyslices[0].max())
            min_z = int(truezslices[0].min())
            max_z = int(truezslices[0].max())
            
            width = (max_x - min_x)/x_length
            height = (max_y - min_y)/y_length
            depth = (max_z - min_z)/z_length
            
            x_center = min_x/x_length + width/2.
            y_center = min_y/y_length + height/2.
            z_center = min_z/z_length + depth/2.   
            
            bbox_list.append([mask_value, z_center, x_center, y_center, depth, width, height])
        
    return bbox_list


def create_txt_label(mask_path: str, label_path: str, label_list: List[int]):
    """
    Creates the MedYOLO txt labels and fills them with the specified label entries.
    """
    nifti_mask = nib.load(mask_path).dataobj
    bbox_list = findBBox(nifti_mask, label_list)    
    
    for bbox in bbox_list:
        mask_class, z_center, x_center, y_center, depth, width, height = bbox
        create_label_3D(z_center, x_center, y_center, depth, width, height, label_path, mask_class)
        

def trainvalsplit(label_list: List[int], datadir: str, savedir: str, mod_number=5, rotate=False):
    """
    Splits a dataset into training and validation sets before generating MedYOLO labels for each set.
    Optional rotation can be applied to the training set.

    This script assumes your NIfTI images are stored in: datadir/img/
    and your NIfTI segmentation masks are store in: datadir/mask/
    It also assumes your images have names like: CT_001_image.nii.gz
    and your segmentations masks have names like: CT_001_segmentation.nii.gz
    """
    # setting the MedYOLO directories
    YOLOtraindatadir = savedir + r"images/train/"
    YOLOtrainlabeldir = savedir + r"labels/train/"
    YOLOvaldatadir = savedir + r"images/val/"
    YOLOvallabeldir = savedir + r"labels/val/"

    # where the original images and corresponding masks are saved
    niftidir = datadir + r"img/"
    maskdir = datadir + r"mask/"
 
    niftilist = []
    for _, _, files in os.walk(niftidir):
        for file in files:
            niftilist.append(file)
            
    random.shuffle(niftilist)
    trainimgs = []
    valimgs = []
    for i in range(len(niftilist)):
        if i % mod_number == 0:
            valimgs.append(niftilist[i])
        else:
            trainimgs.append(niftilist[i])
            
    count = 0
    for file in trainimgs:
        count+=1
        # read paths
        niftipath = niftidir + file
        # using image filename to generate mask filename by removing 'image.nii.gz' from the end of file name
        # and adding 'segmentation.nii.gz' in its place
        maskpath = maskdir + file[:-12] + 'segmentation.nii.gz'
        # write paths
        imgsavepath = YOLOtraindatadir + file
        labelsavepath = YOLOtrainlabeldir + file.split('.')[0] + '.txt'
        
        print(niftipath, ' : ', maskpath, ' ', count, '/', len(trainimgs))
        
        # copy image into train YOLO directory
        shutil.copy(niftipath, imgsavepath)
        # save label for image
        create_txt_label(maskpath, labelsavepath, label_list)
        
        if rotate:
            # make rotated versions of the images and labels
            maskRotate(niftipath, maskpath, imgsavepath, labelsavepath, label_list)
        
        
    count = 0
    for file in valimgs:
        count+=1
        # read paths
        # same operations on filenames as above
        niftipath = niftidir + file
        maskpath = maskdir + file[:-12] + 'segmentation.nii.gz'
        # write paths
        imgsavepath = YOLOvaldatadir + file
        labelsavepath = YOLOvallabeldir + file.split('.')[0] + '.txt'
        
        print(niftipath, ' : ', maskpath, ' ', count, '/', len(valimgs))
        
        # copy image into val YOLO directory
        shutil.copy(niftipath, imgsavepath)
        # save label for image
        create_txt_label(maskpath, labelsavepath, label_list)
        

def maskRotate(imgpath, maskpath, imgsavepath, labelsavepath, label_list):
    """
    Performs rotation on NIfTI images and their segmentation masks,
    then saves out the rotated images and MedYOLO labels generated from the rotated segmentations.
    """
    # read the set of images saved in my training set for the MO YOLO dataset
    # then generate a set of images and bounding boxes with rotation augmention
    from scipy import ndimage
    
    base_rot_angles = [-17., -8., 0., 8., 17.]  # +/- 3.
    
    # load image and mask
    # images in (x, y, z) format
    img = nib.load(imgpath).dataobj
    mask = nib.load(maskpath).dataobj
    
    # rotate and save images and labels
    img_index = 0 # to differentiate between the augmented images and the originals
    for angle in base_rot_angles:
        # set paths to save images
        img_index += 1
        print(img_index)
        rotimgsavepath = imgsavepath.split('.')[0] + '_' + str(img_index) + '.nii.gz'
        rotlabelsavepath = labelsavepath.split('.')[0] + '_' + str(img_index) + '.txt'           
        
        # rotate image and mask
        rot_angle = angle + round(random.uniform(-3., 3.), 1)
        # cval is air in Hounsfield units.  You may want to change it for MR data.
        rot_img = ndimage.rotate(img, rot_angle, reshape=False, cval=-1024.)
        rot_mask = ndimage.rotate(mask, rot_angle, order=0, reshape=False)
    
        # create label for rotated mask
        bbox_list = findBBox(rot_mask, label_list)    
        for bbox in bbox_list:
            mask_class, z_center, x_center, y_center, depth, width, height = bbox
            create_label_3D(z_center, x_center, y_center, depth, width, height, rotlabelsavepath, mask_class)
        
        # save rotated image
        new_image = nib.Nifti1Image(rot_img, np.eye(4))
        nib.save(new_image, rotimgsavepath)
    

if __name__ == '__main__':
    # directory to save the YOLO data to
    YOLOsavedir = r""

    # setting the directory where my nifti files are found
    # /img/ and /masks/ folders inside
    datasourcedir = r""
    
    # segmentation classes to generate labels for
    target_labels = [3, 4, 9, 12]

    # generate training (augmented) and validation data
    trainvalsplit(target_labels, datasourcedir, YOLOsavedir, rotate=True)   
    