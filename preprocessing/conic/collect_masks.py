import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os
import glob

labelspath = ""
outfolder = ""

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

color_dict = {
                1 : [0,0,0], #neutrophil  : black
                2 : [0,255,0], #epithelial : green
                3 : [255,255,0], #lymphocyte : Yellow
                4 : [255,0,0], #plasma : red
                5 : [0,0,255], #eosinophil : Blue
                6 : [255,0,255], #connectivetissue : fuchsia
                0 : [255,255,255] #Background : white
              }

def CreateMaskImage(label_path):
    imname = os.path.split(label_path)[1].split(".")[0]
    label = np.load(label_path)
    label = label[:,:,1]
    k = np.zeros((256, 256, 3))
    l = 256
    for i in range(0,l):
        for j in range(0,l):
            k[i][j] = color_dict[label[i][j]]
    imagepath = os.path.join(outfolder, imname+".png")
    matplotlib.image.imsave(imagepath, k)


label_paths = glob.glob(os.path.join(labelspath,"*.npy"))
for label_path in label_paths:
    CreateMaskImage(label_path)