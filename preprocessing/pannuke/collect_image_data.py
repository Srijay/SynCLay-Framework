import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os

imagepath = "F:/Datasets/PanNuke/Fold2/images_npy/fold2/images.npy"
typespath = "F:/Datasets/PanNuke/Fold2/images_npy/fold2/types.npy"
outfolder = "F:/Datasets/PanNuke/Fold2/images_2/"

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

images = np.load(imagepath)
types = np.load(typespath)
l = len(types)

for i in range(0,l):
    type = types[i]
    imagename = type + "_" + str(i) + ".png"
    imagepath = os.path.join(outfolder,imagename)
    img = images[i]/255.0
    matplotlib.image.imsave(imagepath, img)