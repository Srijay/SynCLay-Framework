import os
import glob
import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import matplotlib
import time
import argparse

PIL.Image.MAX_IMAGE_PIXELS = 933120000

def join_images(input_dir,output_file,height,width):

    start_time = time.time()

    paths = glob.glob(os.path.join(input_dir,"*.png"))

    patch = 256

    image = np.zeros((height,width,3))
    count_masks = np.zeros((height,width,3))
    k=0
    for path in paths:
        imname = os.path.split(path)[1].split(".")[0]
        imname = imname.split("_")
        y,x = int(imname[-2]),int(imname[-1])
        img = Image.open(path)
        img = np.asarray(img)
        #print("X => ",x," Y => ",y)
        image[x:x+patch,y:y+patch,:] += img
        count_masks[x:x+patch,y:y+patch,:]+=1.0
        k+=1

    count_masks = count_masks.clip(min=1)

    image = image/count_masks

    image = image/255.0

    matplotlib.image.imsave(output_file, image)

    print("--- %s seconds ---" % (time.time() - start_time))

    print("Done")

parser = argparse.ArgumentParser()
parser.add_argument("--patches_dir", help="path to generated patches to join",
                    default="F:/Datasets/conic/lizard/sample/valid/cropped/results_safron/pred_image")
parser.add_argument("--output_file", help="path to output file",
                    default="F:/Datasets/conic/lizard/sample/valid/result_safron.jpeg")
parser.add_argument("--im_height", default=1316, type=int, help="image height")
parser.add_argument("--im_width", default=1316, type=int, help="image width")

args = parser.parse_args()

join_images(args.patches_dir,args.output_file,args.im_height,args.im_width)