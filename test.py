import argparse
import torch
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import opt
from places2 import Places2
from places3 import Places3
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from torch.utils import data
import numpy as np
from util.image import unnormalize
from PIL import Image, ImageFilter



parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--snapshot', type=str, default='./model/1000000.pth')
parser.add_argument('--image_size', type=int, default=2048)
args = parser.parse_args(args=[])

device = torch.device('cuda')

size = (1280, 1280)
img_transform = transforms.Compose(
    [  # transforms.CenterCrop((480,480)),
        transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_test = Places2(args.root,args.mask_root, img_transform, mask_transform, 'test')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()



test = iter(data.DataLoader(
    dataset_test, batch_size=1))
y=[]
y_pre=[]



for i in range(len(test)):
    image, mask, gt = [x.to(device) for x in next(test)]
    with torch.no_grad():
        output, _ = model(image, mask)
        #print(output.shape)
    img=gt
    img = img.to(torch.device('cpu'))
    img=unnormalize(img)[0]
    img = np.transpose(img, (1,2,0))
    y.append(img)
    
    img=output
    img = img.to(torch.device('cpu'))
    img=unnormalize(img)[0]
    y_pre.append(img)



for i in range(len(y)):
    plt.figure()
    plt.axis('off')
    plt.imshow(y[i])
    plt.savefig('./testout/image_'+str(i)+'.png', transparent=True)

for i in range(len(y)):
    plt.figure()
    plt.axis('off')
    print(np.shape(y_pre[i]))
    utils.save_image(y_pre[i],'./testout/hr_image'+str(i)+'.png')


