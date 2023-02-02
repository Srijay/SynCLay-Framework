import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os

maskspath = "F:/Datasets/pannuke/fold1/masks_npy/fold1/masks.npy"
outfolder = "F:/Datasets/pannuke/fold1/masks/"
typespath = "F:/Datasets/pannuke/fold1/types.npy"
types = np.load(typespath)

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

color_dict = {
                0 : [0,0,0], #Neoplastic : black
                1 : [0,255,0], #Inflammatory : green
                2 : [255,255,0], #Soft : Yellow
                3 : [255,0,0], #Dead : red
                4 : [0,0,255], #Epithelial dr srijay: Blue
                5 : [255,255,255] #Background : white
              }

def CreateMaskImage(mask,type,index):
    k = np.zeros((256, 256, 3))
    l = 256
    for i in range(0,l):
        for j in range(0,l):
            print(mask[i][j])
            exit(0)
            k[i][j] = color_dict[np.argmax(mask[i][j])]
    imagename = type + "_" + str(index) + ".png"
    imagepath = os.path.join(outfolder, imagename)
    matplotlib.image.imsave(imagepath, k)

masks = np.load(maskspath)
total = len(masks)
for i in range(0,total):
    mask = masks[i]
    type = types[i]
    CreateMaskImage(mask,type,i)