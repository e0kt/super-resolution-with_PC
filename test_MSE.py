import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import opt
from places2 import Places2
from places4 import Places4
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from torch.utils import data
import numpy as np
from util.image import unnormalize
from PIL import Image, ImageFilter
from sklearn.metrics import mean_squared_error




parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--snapshot', type=str, default='./model/1000000.pth')
parser.add_argument('--image_size', type=int, default=2048)
args = parser.parse_args(args=[])

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

img_tf1 = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
img_tf2 = transforms.Compose(
    [ transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


dataset_test =Places4(args.root, args.mask_root, img_tf1,img_tf2, mask_tf, 'test')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()





test = iter(data.DataLoader(
    dataset_test, batch_size=1))
y=[]
y_blur=[]
y_pre=[]
y_num=[]
y_pre_num=[]





for i in range(len(test)):
    image, mask, gt,image2 = [x.to(device) for x in next(test)]
    with torch.no_grad():
        output, _ = model(image2, mask)
    img=gt
    img = img.to(torch.device('cpu'))
    img=unnormalize(img)[0]
    img = np.transpose(img, (1,2,0))
    y.append(img)
    y_num.append(img.numpy())
    
    img=image2
    img = img.to(torch.device('cpu'))
    img=unnormalize(img)[0]
    img = np.transpose(img, (1,2,0))
    y_blur.append(img)
    
    img=output
    img = img.to(torch.device('cpu'))
    img=unnormalize(img)[0]
    img = np.transpose(img, (1,2,0))
    y_pre.append(img)
    y_pre_num.append(img.numpy())





a=img.numpy()





mse=0
for i in range(len(y_num)):
    mse+=sum(sum(sum((y_num[i]-y_pre_num[i])**2)))
mse=mse/len(y_num)
print('MSE:%s'%mse)





for i in range(len(y)):
    plt.figure()
    
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(y[i])
    
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(y_blur[i])
    
    
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(y_pre[i])
    plt.show()

