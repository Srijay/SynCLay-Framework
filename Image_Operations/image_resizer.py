from PIL import Image
import os, sys
import glob
import matplotlib
import matplotlib.pyplot as plt

path = "F:/Datasets/CRAG_LabServer/scene_graph/preprocessing/valid/images"
outdir = "F:/Datasets/CRAG_LabServer/scene_graph/preprocessing/valid/resized_1000/images"
resize_len = 1000

if not os.path.exists(outdir):
        os.makedirs(outdir)

dirs = os.listdir(path)

image_paths = glob.glob(os.path.join(path,"*.png"))

for path in image_paths:
    imname = os.path.split(path)[1]
    savepath = os.path.join(outdir,imname)
    im = Image.open(path)
    imResize = im.resize((resize_len,resize_len), Image.ANTIALIAS)
    #imResize.save(savepath)
    matplotlib.image.imsave(savepath, imResize)