import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os

maskspath = "F:/Datasets/pannuke/fold2/masks_npy/fold2/masks.npy"
outfolder = "F:/Datasets/pannuke/fold2/labels/"
typespath = "F:/Datasets/pannuke/fold2/types.npy"
types = np.load(typespath)


if not os.path.exists(outfolder):
        os.makedirs(outfolder)

label_dict = {
                0 : 1, #Neoplastic : black
                1 : 2, #Inflammatory : green
                2 : 3, #Soft : Yellow
                3 : 4, #Dead : red
                4 : 5, #Epithelial : Blue
                5 : 0 #Background
              }

def CreateLabel(mask,type,index):
    k = np.zeros((256, 256))
    l = 256
    for i in range(0,l):
        for j in range(0,l):
            k[i][j] = label_dict[np.argmax(mask[i][j])]
    imagename = type + "_" + str(index) + ".npy"
    labelpath = os.path.join(outfolder, imagename)
    np.save(labelpath,k)

masks = np.load(maskspath)
total = len(masks)
for i in range(0,total):
    mask = masks[i]
    type = types[i]
    CreateLabel(mask,type,i)