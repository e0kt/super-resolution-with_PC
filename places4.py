import random
import torch
from PIL import Image, ImageFilter
from glob import glob
import copy
import numpy as np
from torchvision import transforms


class Places4(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform1, img_transform2,mask_transform,
                 split='train'):
        super(Places4, self).__init__()
        self.img_transform1 = img_transform1
        self.img_transform2 = img_transform2
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.png'.format(img_root),
                              recursive=True)
            # self.mask_paths = glob('{:s}/train/*.png'.format(mask_root))
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))
            # self.mask_paths = glob('{:s}/test/*.png'.format(mask_root))

        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img=gt_img.convert('RGB')


        gt_img = self.img_transform1(gt_img)
        image_blurred = copy.deepcopy(gt_img) 
        image_blurred = image_blurred.filter(ImageFilter.BLUR)  
        gt_img2 = image_blurred

        gt_img2 = self.img_transform2(gt_img2)

        gt_img = self.img_transform2(gt_img)


        # mask = Image.open(self.mask_paths[index])
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img, gt_img2 * mask

    def __len__(self):
        return len(self.paths)
