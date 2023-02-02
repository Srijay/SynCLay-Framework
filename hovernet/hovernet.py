import glob
import math
import pathlib
import re
from PIL import Image

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision.utils import save_image
import torchvision.transforms as transforms
import json
import math
import multiprocessing
import os
import re
import sys
from importlib import import_module
from multiprocessing import Lock, Pool

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import convert_pytorch_checkpoint

from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)

from utils import remove_alpha_channel


class InferManager():
    """Run inference on tiles."""

    def __init__(self, model_path, mode, type_info_path):

        self.run_step = None
        self.mode = mode
        self.model_path = model_path
        self.type_info_path = type_info_path

        if self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.nr_types = len(self.type_info_dict)

        #Load the pretrained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_desc = import_module("net_desc")
        model_creator = getattr(model_desc, "create_model")
        net = model_creator(mode=self.mode, nr_types=self.nr_types)
        net = net.to(device)

        saved_state_dict = torch.load(self.model_path)["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
        net.load_state_dict(saved_state_dict)
        net = torch.nn.DataParallel(net)

        # torch.save(net.state_dict(), "conic_weights.pth", _use_new_zipfile_serialization=False)
        # exit(0)

        module_lib = import_module("run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_step = lambda input_batch: run_step(input_batch, net)

        self.net = net

        return


    def tensor_to_patches(self,image,centre_crop=0):
        padl = 46
        padr = 328
        padt = 46
        padb = 328
        patch_input_shape = 256
        patch_output_shape = 164

        #img = np.lib.pad(image.cpu().numpy(), ((padt, padb), (padl, padr), (0, 0)), "constant")
        img = torch.nn.functional.pad(image, (0, 0, padl, padr, padt, padb), "constant")

        patch_indices = [(0, 0), (164, 0), (0, 164), (164, 164)]
        patches = []

        for patch_index in patch_indices:
            patch_data = img[
                         patch_index[0]: patch_index[0] + patch_input_shape,
                         patch_index[1]: patch_index[1] + patch_input_shape,
                         ]
            sample = patch_data

            if(centre_crop): #Only for labels
                sample = patch_data.permute(2,0,1)
                transform = transforms.CenterCrop((164, 164))
                sample = transform(sample)
                sample = sample.squeeze()

            # save_image(sample.permute(2, 0, 1) / 255.0, "test_4_image" + str(k) + ".png")
            # k+=1
            patches.append(sample)

        patches = torch.stack(patches)
        return patches


    def get_segmentation(self, images):
        patches = self.tensor_to_patches(images[0])
        class_outputs = self.run_step(patches)
        return class_outputs



if __name__ == '__main__':

    label_path = "F:/Datasets/conic/CoNIC_Challenge/challenge_trial/train/sample/challenge_labels/1.npy"
    label = np.load(label_path)
    class_label = label[:,:,1]
    class_label = class_label.astype(np.int32)
    class_label_tensor = torch.from_numpy(class_label)
    class_label_tensor = class_label_tensor.cuda()
    class_label_tensor = class_label_tensor[:,:,None]
    print(class_label_tensor.shape)
    batch_size = 1
    nr_types = 7
    input_dir = "F:/Datasets/conic/CoNIC_Challenge/challenge_trial/train/sample/masks"
    output_dir = "F:/Datasets/conic/CoNIC_Challenge/challenge_trial/train/sample/hovernet_output"

    # * depend on the number of samples and their size, this may be less efficient
    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    file_path_list = glob.glob(patterning("%s/*" % input_dir))
    file_path_list.sort()  # ensure same order
    assert len(file_path_list) > 0, 'Not Detected Any Files From Path'

    rm_n_mkdir(output_dir + '/overlay/')
    rm_n_mkdir(output_dir + '/segmentation/')
    rm_n_mkdir(output_dir + '/color_segmentation/')

    images_np = []
    image_names = []

    while len(file_path_list) > 0:
        file_path = file_path_list.pop(0)
        base_name = pathlib.Path(file_path).stem
        image_names.append(base_name)
        img = np.array(Image.open(file_path))
        img = remove_alpha_channel(img)
        images_np.append(img)

    images_np = np.array(images_np)
    images_np = torch.from_numpy(images_np)
    images_np = images_np.cuda()

    infer = InferManager(model_path="conic.tar",
                         mode="fast",
                         nr_types=7)

    #class_result = infer.tensor_to_patches(class_label_tensor,centre_crop=164)
    img_result = infer.get_segmentation(images_np)

    print(class_result.shape)
    print(img_result.shape)


