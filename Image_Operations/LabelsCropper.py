import glob
import os
from PIL import Image
import numpy as np
import PIL

images_input_folder = "F:/Datasets/conic/lizard/sample/valid/inst_labels"

images_output_folder = "F:/Datasets/conic/lizard/sample/valid/cropped/inst_labels"


if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

patch_size = 288
stride = 212

def CropImage(path):

    image_im = np.load(path)

    image_im = Image.fromarray(image_im)

    image_initial = os.path.split(path)[1].split('.')[0]

    width, height = image_im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):

        while (right < width):

            left = x
            top = y
            right = left + patch_size
            bottom = top + patch_size
            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset

            im_crop_name = image_initial + "_" + str(x) + "_" + str(y) + ".npy"

            im_crop_image = image_im.crop((left, top, right, bottom))

            output_image_path = os.path.join(images_output_folder, im_crop_name)

            np.save(output_image_path, np.asarray(im_crop_image))

            x += stride

        x = 0
        right = 0
        y += stride

    return

avgs = []
image_paths = glob.glob(os.path.join(images_input_folder,"*.npy"))
image_names = []
for path in image_paths:
    mbt_index = CropImage(path)
#print(np.max(avgs))
#print(np.mean(avgs))

