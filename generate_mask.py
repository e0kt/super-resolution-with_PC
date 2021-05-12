import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os


image_size=512  

fileList=os.listdir('data/test_large')
for num in range(len(fileList)):
    image=plt.imread('data/test_large/'+fileList[num])
    mask = np.zeros((image_size, image_size, 3), dtype='int')
    mask[:] = 0
    for i in range(len(image)):
        for j in range(len(image[0])):
            col=int(image_size/len(image)*i)
            rol=int(image_size/len(image[0])*j)
            if col<image_size and rol<image_size:
                mask[col,rol,:]=255

    image2 = Image.fromarray(np.uint8(mask))  
    image2.save('./masks/test/'+str(num)+'.png')
