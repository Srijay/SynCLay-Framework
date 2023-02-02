import numpy as np
import matplotlib
import os
import glob
from tensorflow.keras.preprocessing import image
import PIL
import matplotlib.pyplot as plt

PIL.Image.MAX_IMAGE_PIXELS = 933120000

masks_folder = "F:/Datasets/DigestPath/scene_generation/all/1000/train_data/masks"
images_folder = "F:/Datasets/DigestPath/scene_generation/all/1000/train_data/images"
outfolder = "F:/Datasets/DigestPath/scene_generation/all/1000/train_data/tri_masks_3diffcolors_new"
outfolder_images = "F:/Datasets/DigestPath/scene_generation/raw/images_new"

masks_paths = glob.glob(os.path.join(masks_folder,"*.png"))

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

if not os.path.exists(outfolder_images):
        os.makedirs(outfolder_images)

def extract_image(mask_path):
    mask_name = os.path.split(mask_path)[1]
    mask_name = mask_name.split(".")[0]+".png"
    img_name = mask_name.split(".")[0]+".png"
    img_path = os.path.join(images_folder,img_name)
    mask = image.load_img(mask_path)
    img = image.load_img(img_path)
    mask_np = image.img_to_array(mask)
    image_np = image.img_to_array(img)
    if mask_np.shape[2] == 4:
        mask_np = mask_np[:,:,:3]
    if image_np.shape[2] == 4:
        image_np = image_np[:,:,:3]

    mask_axis_max = np.max(mask_np,axis=2)
    mask_np[:, :, 0] = mask_axis_max
    mask_np[:, :, 1] = mask_axis_max
    mask_np[:, :, 2] = mask_axis_max
    mask_np[mask_np < 100] = 0 #tissue + white portion
    mask_np[mask_np >= 100] = 2 #glands

    image_axis_min = np.mean(image_np, axis=2)
    image_np[:, :, 0] = image_axis_min
    image_np[:, :, 1] = image_axis_min
    image_np[:, :, 2] = image_axis_min
    image_np[image_np >= 230] = 255 #white portion -> 1 tissue (after inversion)
    image_np[image_np < 230] = 0 #tissue -> 0 white portion (after inversion)
    image_np=image_np/255
    image_np = 1 - image_np

    new_mk = mask_np+image_np

    new_mk[:, :, 0][new_mk[:, :, 0]==0] = 0 #white background : Blue color
    new_mk[:, :, 1][new_mk[:, :, 1]==0] = 0 #white background : Blue color
    new_mk[:, :, 2][new_mk[:, :, 2]==0] = 255 #white background : Blue color

    new_mk[:, :, 0][new_mk[:, :, 0] == 1] = 255  # tissue region : Red color
    new_mk[:, :, 1][new_mk[:, :, 1] == 1] = 0  # tissue region : Red color
    new_mk[:, :, 2][new_mk[:, :, 2] == 1] = 0  # tissue region : Red color

    new_mk[:, :, 0][new_mk[:, :, 0] == 2] = 0  # Glands : Green color
    new_mk[:, :, 1][new_mk[:, :, 1] == 2] = 255  # Glands : Green color
    new_mk[:, :, 2][new_mk[:, :, 2] == 2] = 0  # Glands : Green color

    new_mk[:, :, 0][new_mk[:, :, 0] == 3] = 0  # Glands : Green color
    new_mk[:, :, 1][new_mk[:, :, 1] == 3] = 255  # Glands : Green color
    new_mk[:, :, 2][new_mk[:, :, 2] == 3] = 0  # Glands : Green color

    #new_mk = new_mk/255.0
    #plt.imshow(new_mk)
    #plt.show()
    print(mask_name)

    image.save_img(os.path.join(outfolder,mask_name),new_mk)
    #image.save_img(os.path.join(outfolder_images,img_name),img)

for path in masks_paths:
    if("neg" in path):
        print(path)
        extract_image(path)