import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 2M image in DIV2K dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.png'.format(img_root),
                              recursive=True)
            self.mask_paths = glob('{:s}/train/*.png'.format(mask_root))

        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))
            self.mask_paths = glob('{:s}/test/*.png'.format(mask_root))


        # self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[index])
        # mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
