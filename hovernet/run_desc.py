import torch.nn.functional as F
import cv2
from utils import *
from torchvision.utils import save_image
from collections import OrderedDict
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
import os

from skimage.segmentation import watershed

####
def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.permute(0, 3, 1, 2).contiguous()
    patch_imgs_gpu = patch_imgs_gpu/255.0
    pred_dict = model(patch_imgs_gpu)

    # inst_label = np.argmax(pred_dict["tp"].cpu().numpy(), axis=1)
    # inst_label[inst_label>0] = 1.0
    # inst_label = __proc_np_hv(inst_label)
    # for i in range(4):
    #     plt.imsave('object_mask_' + str(i) + '.png', inst_label[i])
    #
    # print(inst_label.shape)
    # print("Won")
    # exit(0)

    class_label = pred_dict["tp"]
    return class_label


def __proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb[blb > 0] = 1  # background is 0 already


    dist = (1.0 - 0) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    marker = blb
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred

####
from itertools import chain
