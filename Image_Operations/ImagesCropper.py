import glob
import os
from PIL import Image
import numpy as np
import PIL

folder_path = "F:/Datasets/BCSS"
masks_input_folder = os.path.join(folder_path, "masks")
images_input_folder = os.path.join(folder_path, "images")

output_dir = "F:/Datasets/BCSS/cropped/1024"
masks_output_folder = os.path.join(output_dir, "masks")
images_output_folder = os.path.join(output_dir, "images")

if not os.path.exists(masks_output_folder):
        os.makedirs(masks_output_folder)
if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

patch_size = 1024
stride = 512
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def CropImage(imgname):
    masks_image_path = os.path.join(masks_input_folder,imgname+".png")
    images_image_path = os.path.join(images_input_folder, imgname+".2500.png")
    image_initial = imgname

    mask_im = Image.open(masks_image_path)

    image_im = Image.open(images_image_path)


    width, height = mask_im.size

    # if mask_im.shape[2] == 4:
    #     mask_im = mask_im[:,:,:3]
    # if image_im.shape[2] == 4:
    #     image_im = image_im[:,:,:3]

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

            im_crop_name = image_initial + "_" + str(x) + "_" + str(y) + ".png"

            im_crop_mask = mask_im.crop((left, top, right, bottom))

            im_crop_image = image_im.crop((left, top, right, bottom))

            output_mask_path = os.path.join(masks_output_folder, im_crop_name)
            output_image_path = os.path.join(images_output_folder, im_crop_name)
            im_crop_mask.save(output_mask_path)
            im_crop_image.save(output_image_path)

            x += stride

        x = 0
        right = 0
        y += stride

avgs = []
masks_image_paths = glob.glob(os.path.join(masks_input_folder,"*.png"))
image_names = []
for path in masks_image_paths:
    image_names.append(os.path.split(path)[1].split('.')[0])
for imgname in image_names:
    print(imgname)
    CropImage(imgname)

avgs = np.array(avgs)
#print(np.max(avgs))
#print(np.mean(avgs))
