import argparse
import json
import math
import os
import sys
from datetime import datetime

import cv2, json, pickle
import matplotlib
import numpy as np
import torch
from torchvision.utils import save_image

sys.path.insert(0,'./../')
sys.path.insert(0,'./../hovernet')

from model import Sg2ImModel
from scipy.spatial import distance

from image_from_scene_graph import hovernet_class_output_to_class_image, colored_images_from_classes
from utils import mkdir, save_numpy_image_FLOAT

from preprocessing.delaunay_triangulation_nuclei import vertex, delaunay_triangulation, draw_point

def int_tuple(s):
  return tuple(int(i) for i in s.split(','))

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='./../outputs/conic_residual_2/model/model_hovernet_30epochs.pt')
parser.add_argument('--config', default='./../outputs/conic_residual_2/config.txt')
parser.add_argument('--output_dir', default='outputs/gui')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])
parser.add_argument('--type_info_path', default='./../hovernet/type_info/conic.json')
parser.add_argument('--hovernet_model_path', default='./../hovernet/trained_models/conic.tar')
parser.add_argument('--image_size', default='256,256', type=int_tuple)

#Object vector parameters
parser.add_argument('--use_size_feature', default=0, type=int)
parser.add_argument('--use_loc_feature', default=1, type=int)

args = parser.parse_args()

VOCAB = {"neutrophil": 0, "epithelial": 1, "lymphocyte": 2, "plasma": 3, "eosinophil": 4, "connectivetissue": 5}

cells_size_distribution_file = './../data/thecot_model/cells_size_distributions.obj'
cells_size_distributions = pickle.load(open(cells_size_distribution_file, 'rb'))

color_dict = {
        "neutrophil": [0, 0, 0],  # neutrophil  : black
        "epithelial": [0, 255, 0],  # epithelial : green
        "lymphocyte": [255, 165, 0],  # lymphocyte : Yellow
        "plasma": [255, 0, 0],  # plasma : red
        "eosinophil": [0, 0, 255],  # eosinophil : Blue
        "connectivetissue": [255, 0, 255],  # connectivetissue : fuchsia
    }


def get_model():
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    output_dir = os.path.join('images', args.output_dir)
    if not os.path.isdir(output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(output_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    with open(args.config) as f:
        config = json.load(f)

    checkpoint['model_kwargs']['integrate_hovernet'] = config['integrate_hovernet']
    checkpoint['model_kwargs']['gpu_dict'] = gpu_dict = json.loads(config['gpu_model_dict'])
    checkpoint['model_kwargs']['type_info_path'] = args.type_info_path
    checkpoint['model_kwargs']['hovernet_model_path'] = args.hovernet_model_path
    model = Sg2ImModel(**checkpoint['model_kwargs'])
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    print("Model loaded successfully")
    model.eval()
    model.to(device)
    return model, gpu_dict


def json_to_img(scene_graph, model, gpu_dict):

    object_embeddings, triples, objects_boxes_gt, graph_image = json_to_scene_graph(scene_graph)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # Run the model forward
    with torch.no_grad():

        if torch.cuda.is_available():

            object_embeddings = object_embeddings.cuda()
            triples = triples.cuda()
            objects_boxes_gt = objects_boxes_gt.cuda()
            image_pred, mask_pred, label_pred_hovernet_patches, object_masks_pred, object_boxes, label_gt_hovernet_patches = model(object_indices=None,
                                                                                                                                    mask=None,
                                                                                                                                    object_coordinates=None,
                                                                                                                                    object_embeddings=object_embeddings,
                                                                                                                                    triples=triples,
                                                                                                                                    objects_boxes_gt=objects_boxes_gt,
                                                                                                                                    gpu_dict=gpu_dict)


            if(image_pred is None):
                print("No image is constructed")
                return None,None,None

            label_pred_hovernet_class_output = np.argmax(label_pred_hovernet_patches.cpu().numpy(), axis=1)
            label_pred = hovernet_class_output_to_class_image(label_pred_hovernet_class_output, image_size=args.image_size[0])
            label_pred_img = colored_images_from_classes(args, label_pred)

    image_output_dir = os.path.join(args.output_dir,'images')
    mask_output_dir = os.path.join(args.output_dir,'masks')
    graph_output_dir = os.path.join(args.output_dir,'graphs')

    mkdir(image_output_dir)
    mkdir(mask_output_dir)
    mkdir(graph_output_dir)

    #save graph
    graph_path = os.path.join(graph_output_dir, 'graph{}.png'.format(current_time))
    save_numpy_image_FLOAT(graph_image, graph_path)

    # Save the component mask layout and generated image
    mask_path = os.path.join(mask_output_dir, 'mask{}.png'.format(current_time))
    save_numpy_image_FLOAT(label_pred_img, mask_path)
    # save_image(image_pred[0], mask_path)

    img_path = os.path.join(image_output_dir, 'image{}.png'.format(current_time))
    save_image(image_pred[0], img_path)

    return graph_path, img_path, mask_path


def one_hot_to_rgb(one_hot, colors):
    one_hot_3d = torch.einsum('abcd,be->aecd', (one_hot.cpu(), colors.cpu()))
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def create_embedding(object_name, object_location, coarse_bbox=None):
    embedding = []

    class_vector = [0.0] * len(VOCAB)
    class_vector[VOCAB[object_name]] = 1.0
    embedding = embedding + class_vector

    if (args.use_loc_feature):
        location_vector = [0.0] * 2  # vector of size 2 for location
        H, W = args.image_size
        location_vector[0] = object_location[0] / W
        location_vector[1] = object_location[1] / H  # Scaling the locations
        embedding = embedding + location_vector

    if (args.use_size_feature):  # Size is determined by area of gland
        size_embedding = [0.0] * 10
        max_area = args.image_size[0] * args.image_size[0] * 1.0
        xsize = coarse_bbox[1] - coarse_bbox[0]
        ysize = coarse_bbox[3] - coarse_bbox[2]
        object_bbox_area = xsize * ysize
        size_index = int(object_bbox_area * 10.0 / max_area)
        if (size_index > 9):
            size_index = 9
        size_embedding[size_index] = 1
        embedding = embedding + size_embedding

    return embedding


def json_to_scene_graph(json_text):

    def adjust_range(point_coord):
        if(point_coord<0):
            point_coord = 0.0
        if(point_coord>=1):
            point_coord = 1.0
        return point_coord

    scene = json.loads(json_text)

    if len(scene) == 0:
        print("empty scene graph")
        return []

    scene = scene['objects']
    objects = [i['text'].lower() for i in scene]

    image_size = args.image_size[0]

    object_embeddings = []
    triples = []
    objects_boxes_gt = []
    vertex_dict_int = {}
    vertices_int = []
    points = []
    object_id_to_index = {}
    index = 0
    object_names = []

    hardcoded_benign_bounding_boxes = [(0.02734375, 0.01171875), (0.02734375, 0.0234375), (0.03515625, 0.0234375), (0.03125, 0.03515625), (0.01953125, 0.01953125), (0.03125, 0.03515625), (0.03125, 0.03125), (0.02734375, 0.0390625), (0.0390625, 0.05078125), (0.03515625, 0.0546875), (0.0234375, 0.04296875), (0.046875, 0.05078125), (0.0234375, 0.0546875), (0.0234375, 0.03125), (0.03515625, 0.04296875), (0.046875, 0.046875), (0.0390625, 0.03515625), (0.02734375, 0.02734375), (0.03125, 0.03515625), (0.0234375, 0.0234375), (0.02734375, 0.03125), (0.02734375, 0.0546875), (0.03515625, 0.03125), (0.03125, 0.03125), (0.03515625, 0.03125), (0.03515625, 0.03125), (0.04296875, 0.0625), (0.02734375, 0.03125), (0.05078125, 0.0703125), (0.02734375, 0.03125), (0.03125, 0.0390625), (0.03125, 0.02734375), (0.03515625, 0.03125), (0.03515625, 0.03125), (0.03125, 0.03125), (0.02734375, 0.03515625), (0.0390625, 0.03515625), (0.0390625, 0.03515625), (0.0234375, 0.03515625), (0.0703125, 0.0390625), (0.04296875, 0.0390625), (0.05078125, 0.04296875), (0.04296875, 0.03515625), (0.03125, 0.03515625), (0.046875, 0.05078125), (0.0546875, 0.046875), (0.046875, 0.05078125), (0.046875, 0.046875), (0.046875, 0.03515625), (0.04296875, 0.0390625), (0.046875, 0.046875), (0.046875, 0.046875), (0.0546875, 0.03125), (0.046875, 0.04296875), (0.0390625, 0.03515625), (0.05859375, 0.046875), (0.0390625, 0.03125), (0.05078125, 0.05078125), (0.04296875, 0.04296875), (0.0625, 0.05859375), (0.05859375, 0.0546875), (0.05078125, 0.046875), (0.05859375, 0.0546875), (0.0546875, 0.04296875), (0.04296875, 0.04296875), (0.04296875, 0.0390625), (0.02734375, 0.03125), (0.0390625, 0.05859375), (0.0390625, 0.0703125), (0.0390625, 0.0546875), (0.03125, 0.05859375), (0.0234375, 0.04296875), (0.02734375, 0.0625), (0.02734375, 0.04296875), (0.03515625, 0.05078125), (0.0390625, 0.046875), (0.03515625, 0.06640625), (0.04296875, 0.046875), (0.03515625, 0.046875), (0.0546875, 0.046875), (0.0703125, 0.046875), (0.05859375, 0.03515625), (0.02734375, 0.06640625), (0.0390625, 0.05859375), (0.05078125, 0.05078125), (0.0703125, 0.046875), (0.06640625, 0.046875), (0.078125, 0.03515625), (0.03125, 0.03515625), (0.0234375, 0.0078125), (0.046875, 0.015625), (0.03515625, 0.01953125), (0.046875, 0.0390625), (0.03515625, 0.03125), (0.04296875, 0.03515625), (0.03125, 0.02734375), (0.01171875, 0.04296875), (0.03515625, 0.046875), (0.0234375, 0.0234375), (0.04296875, 0.03125), (0.0390625, 0.05078125), (0.03125, 0.03125), (0.02734375, 0.05078125), (0.02734375, 0.03515625), (0.03125, 0.0390625), (0.0390625, 0.0390625), (0.06640625, 0.03125), (0.0234375, 0.02734375), (0.03515625, 0.046875), (0.05078125, 0.05078125), (0.03515625, 0.0390625), (0.05859375, 0.046875), (0.01953125, 0.03125), (0.02734375, 0.046875), (0.03125, 0.03125), (0.0390625, 0.04296875), (0.046875, 0.0390625), (0.05078125, 0.04296875), (0.0390625, 0.03125), (0.0546875, 0.04296875), (0.03515625, 0.02734375), (0.0390625, 0.046875), (0.0390625, 0.05078125), (0.0390625, 0.03125), (0.0234375, 0.0078125), (0.01953125, 0.01953125), (0.0390625, 0.0546875), (0.03515625, 0.05078125), (0.03125, 0.04296875), (0.03125, 0.03515625), (0.03515625, 0.04296875), (0.01953125, 0.0546875)]
    hardcoded_malignant_bounding_boxes = [(0.04296875, 0.0546875), (0.046875, 0.0390625), (0.05078125, 0.05859375), (0.0546875, 0.078125), (0.0546875, 0.078125), (0.03515625, 0.02734375), (0.05078125, 0.05078125), (0.03125, 0.05078125), (0.0234375, 0.02734375), (0.0390625, 0.04296875), (0.0390625, 0.03515625), (0.03125, 0.0234375), (0.0546875, 0.04296875), (0.04296875, 0.0390625), (0.04296875, 0.03125), (0.015625, 0.04296875), (0.03515625, 0.04296875), (0.03125, 0.03125), (0.03515625, 0.03515625), (0.03125, 0.03125), (0.0234375, 0.03125), (0.04296875, 0.046875), (0.05859375, 0.02734375), (0.046875, 0.0390625), (0.05859375, 0.04296875), (0.0703125, 0.046875), (0.05078125, 0.05078125), (0.05859375, 0.0546875), (0.04296875, 0.0546875), (0.0390625, 0.06640625), (0.05859375, 0.06640625), (0.07421875, 0.04296875), (0.0625, 0.03515625), (0.05859375, 0.0390625), (0.0546875, 0.0390625), (0.0546875, 0.0390625), (0.0625, 0.04296875), (0.06640625, 0.0390625), (0.046875, 0.0390625), (0.0546875, 0.05078125), (0.0390625, 0.046875), (0.0390625, 0.046875), (0.04296875, 0.0546875), (0.0625, 0.06640625), (0.0390625, 0.0625), (0.05078125, 0.06640625), (0.07421875, 0.0546875), (0.0625, 0.05859375), (0.05078125, 0.0390625), (0.05078125, 0.04296875), (0.0625, 0.05859375), (0.03125, 0.03125), (0.046875, 0.046875), (0.04296875, 0.046875), (0.046875, 0.0625), (0.05078125, 0.04296875), (0.05078125, 0.04296875), (0.05078125, 0.0390625), (0.06640625, 0.046875), (0.0546875, 0.046875), (0.06640625, 0.04296875), (0.08984375, 0.0390625), (0.04296875, 0.0390625), (0.05078125, 0.046875), (0.05078125, 0.05859375), (0.046875, 0.0078125), (0.0625, 0.08984375), (0.0390625, 0.10546875), (0.03515625, 0.05078125), (0.03125, 0.0546875), (0.02734375, 0.0546875), (0.04296875, 0.03515625), (0.03515625, 0.03125), (0.03125, 0.0234375), (0.05859375, 0.046875), (0.03515625, 0.02734375), (0.046875, 0.02734375), (0.0625, 0.03125), (0.046875, 0.03125), (0.04296875, 0.04296875), (0.03125, 0.04296875), (0.0390625, 0.0625), (0.05859375, 0.04296875), (0.0390625, 0.0390625), (0.04296875, 0.046875), (0.0625, 0.03515625), (0.0625, 0.02734375), (0.0703125, 0.05078125), (0.046875, 0.0625), (0.05078125, 0.05859375), (0.0078125, 0.0234375), (0.0546875, 0.0390625), (0.05078125, 0.03515625), (0.07421875, 0.04296875), (0.03125, 0.04296875), (0.046875, 0.046875), (0.05859375, 0.046875), (0.0390625, 0.0390625), (0.03515625, 0.0390625), (0.01953125, 0.046875), (0.05078125, 0.03125), (0.04296875, 0.0390625), (0.0390625, 0.02734375), (0.0234375, 0.0390625), (0.04296875, 0.0625), (0.0703125, 0.0625), (0.05859375, 0.046875), (0.03125, 0.03515625), (0.04296875, 0.04296875), (0.03125, 0.05078125), (0.05078125, 0.078125), (0.046875, 0.03125), (0.0390625, 0.0390625), (0.03125, 0.03125), (0.03125, 0.0390625), (0.01171875, 0.03515625), (0.03515625, 0.03515625), (0.05078125, 0.0390625), (0.03125, 0.05078125), (0.046875, 0.03515625), (0.04296875, 0.03515625), (0.03515625, 0.0390625), (0.01953125, 0.02734375), (0.05859375, 0.05078125), (0.0703125, 0.05078125), (0.05859375, 0.04296875), (0.0703125, 0.03515625), (0.06640625, 0.02734375), (0.07421875, 0.06640625), (0.0390625, 0.0390625), (0.0390625, 0.04296875), (0.046875, 0.04296875), (0.04296875, 0.0390625), (0.046875, 0.046875), (0.046875, 0.08203125), (0.03515625, 0.0390625), (0.015625, 0.02734375), (0.0390625, 0.03515625), (0.05078125, 0.046875), (0.04296875, 0.05078125), (0.03125, 0.02734375), (0.0546875, 0.0546875), (0.0390625, 0.02734375), (0.02734375, 0.03515625), (0.03515625, 0.0390625), (0.03125, 0.03125), (0.03515625, 0.0390625), (0.0390625, 0.046875), (0.0390625, 0.05078125), (0.05078125, 0.01953125), (0.078125, 0.0625), (0.06640625, 0.04296875), (0.0390625, 0.046875), (0.1015625, 0.02734375), (0.0703125, 0.0390625), (0.06640625, 0.04296875), (0.07421875, 0.0234375), (0.078125, 0.046875), (0.046875, 0.046875), (0.046875, 0.0390625), (0.03515625, 0.03125), (0.046875, 0.046875), (0.046875, 0.05078125), (0.046875, 0.05078125), (0.0390625, 0.02734375), (0.02734375, 0.03125), (0.02734375, 0.02734375), (0.0234375, 0.0234375), (0.02734375, 0.07421875), (0.01953125, 0.0234375), (0.0390625, 0.03125), (0.0078125, 0.03125), (0.03125, 0.03515625), (0.03515625, 0.02734375), (0.03125, 0.04296875), (0.04296875, 0.01171875), (0.03515625, 0.02734375), (0.04296875, 0.03515625), (0.02734375, 0.0390625), (0.03125, 0.03515625), (0.0390625, 0.0390625), (0.046875, 0.04296875), (0.05078125, 0.04296875), (0.05078125, 0.03515625), (0.046875, 0.0625), (0.03515625, 0.03125), (0.046875, 0.03125), (0.01953125, 0.0078125), (0.03125, 0.03125), (0.046875, 0.03125), (0.03125, 0.05078125), (0.03515625, 0.0234375)]
    # hardcoded_malignant_bounding_boxes = [(0.04296875, 0.0546875), (0.046875, 0.0390625), (0.05078125, 0.05859375), (0.0546875, 0.078125), (0.0546875, 0.078125), (0.03515625, 0.02734375), (0.05078125, 0.05078125), (0.03125, 0.05078125), (0.0390625, 0.04296875), (0.0390625, 0.03515625), (0.03125,0.0234375), (0.0546875, 0.04296875), (0.04296875, 0.0390625), (0.04296875, 0.03125), (0.03515625, 0.04296875), (0.03515625, 0.04296875), (0.03125, 0.03125), (0.03515625, 0.03515625), (0.03125, 0.03125), (0.0234375, 0.03125), (0.04296875, 0.046875), (0.05859375, 0.02734375), (0.046875, 0.0390625), (0.05859375, 0.04296875), (0.0703125, 0.046875), (0.05078125, 0.05078125), (0.05859375, 0.0546875), (0.04296875, 0.0546875), (0.0390625, 0.06640625), (0.05859375, 0.06640625), (0.07421875, 0.04296875), (0.0625, 0.03515625), (0.05859375, 0.0390625), (0.0546875, 0.0390625), (0.0546875, 0.0390625), (0.0625, 0.04296875), (0.06640625, 0.0390625), (0.046875, 0.0390625), (0.0546875, 0.05078125), (0.0390625, 0.046875), (0.0390625, 0.046875), (0.04296875, 0.0546875), (0.0625, 0.06640625), (0.0390625, 0.0625), (0.05078125, 0.06640625), (0.07421875, 0.0546875), (0.0625, 0.05859375), (0.05078125, 0.0390625), (0.05078125, 0.04296875), (0.0625, 0.05859375), (0.03125, 0.03125), (0.046875, 0.046875), (0.04296875, 0.046875), (0.046875, 0.0625), (0.05078125, 0.04296875), (0.05078125, 0.04296875), (0.05078125, 0.0390625), (0.06640625, 0.046875), (0.0546875, 0.046875), (0.06640625, 0.04296875), (0.08984375, 0.0390625), (0.04296875, 0.0390625), (0.05078125, 0.046875), (0.05078125, 0.05859375), (0.046875, 0.0078125), (0.0625, 0.08984375), (0.0390625, 0.10546875), (0.03515625, 0.05078125), (0.03125, 0.0546875), (0.02734375, 0.0546875), (0.04296875, 0.03515625), (0.03515625, 0.03125), (0.03125, 0.0234375), (0.05859375, 0.046875), (0.03515625,0.02734375), (0.046875, 0.02734375), (0.0625, 0.03125), (0.046875, 0.03125), (0.04296875, 0.04296875), (0.03125, 0.04296875), (0.0390625, 0.0625), (0.05859375, 0.04296875), (0.0390625, 0.0390625), (0.04296875, 0.046875), (0.0625, 0.03515625), (0.0625, 0.02734375), (0.0703125, 0.05078125), (0.046875, 0.0625), (0.05078125, 0.05859375), (0.03515625, 0.04296875), (0.0546875, 0.0390625), (0.05078125, 0.03515625), (0.07421875, 0.04296875), (0.03125, 0.04296875), (0.0546875, 0.046875), (0.05859375, 0.046875), (0.0390625, 0.0390625), (0.03515625, 0.0390625), (0.05078125, 0.03125), (0.04296875, 0.0390625), (0.0390625, 0.02734375), (0.04296875, 0.0625), (0.0703125, 0.0625), (0.05859375, 0.046875), (0.03125, 0.03515625), (0.04296875, 0.04296875), (0.05078125, 0.078125), (0.03125, 0.078125), (0.046875, 0.03125), (0.0390625, 0.0390625), (0.03125, 0.03125), (0.03125, 0.0390625), (0.03515625, 0.03515625), (0.05078125, 0.0390625), (0.03125, 0.05078125), (0.046875, 0.03515625), (0.015625, 0.0234375), (0.03515625, 0.0390625),(0.05859375, 0.05078125), (0.0703125, 0.05078125), (0.05859375, 0.04296875), (0.0703125, 0.03515625), (0.06640625, 0.02734375),(0.07421875, 0.06640625), (0.0390625, 0.0390625), (0.02734375, 0.04296875), (0.046875, 0.04296875), (0.04296875, 0.0390625), (0.046875, 0.046875), (0.046875, 0.08203125), (0.03515625, 0.0390625), (0.0390625, 0.03515625), (0.05078125, 0.046875), (0.03125,0.05078125), (0.03125, 0.02734375), (0.0546875, 0.0546875), (0.0390625, 0.02734375), (0.03515625, 0.0390625), (0.03515625, 0.0390625), (0.01171875, 0.0390625), (0.0390625, 0.05078125), (0.046875, 0.01953125), (0.078125, 0.0625), (0.06640625, 0.04296875), (0.0390625, 0.046875), (0.1015625, 0.02734375), (0.0703125, 0.0390625), (0.06640625, 0.04296875), (0.07421875, 0.0234375), (0.078125, 0.046875), (0.046875, 0.046875), (0.046875, 0.0390625), (0.03515625, 0.03125), (0.046875, 0.046875), (0.046875, 0.05078125), (0.046875, 0.05078125), (0.0390625, 0.02734375), (0.02734375, 0.03125), (0.02734375, 0.02734375), (0.0234375, 0.0234375), (0.01953125, 0.0234375), (0.0390625, 0.03125), (0.03125, 0.03515625), (0.03515625, 0.02734375), (0.03125, 0.04296875), (0.04296875,0.01171875), (0.03515625, 0.02734375), (0.04296875, 0.03515625), (0.02734375, 0.0390625), (0.03125, 0.03515625), (0.0390625, 0.0390625), (0.046875, 0.04296875), (0.05078125, 0.04296875), (0.05078125, 0.03515625), (0.046875, 0.0625), (0.03515625, 0.03125),(0.046875, 0.03125), (0.01953125, 0.0078125), (0.03125, 0.03125), (0.046875, 0.03125), (0.03125, 0.05078125), (0.03515625, 0.0234375)]

    for i in range(0, len(objects)):

        obj_s = scene[i]

        # build vertex info
        sx0 = obj_s['left']
        sy0 = obj_s['top']

        point = (sx0,sy0)

        if (point[0] < 1 and point[1] < 1):
            point = (adjust_range(point[0]), adjust_range(point[1]))
            point = (int(point[0] * image_size), int(point[1] * image_size))

        if (point[0] == 256):
            point = (255, point[1])
        if (point[1] == 256):
            point = (point[0], 255)

        points.append(point)

        object_name = obj_s['text'].lower()
        object_names.append(object_name)

        cell_stats = cells_size_distributions[VOCAB[object_name]]

        # sampled_width = np.random.normal(cell_stats[0][0], cell_stats[0][1], 1)
        # sampled_height = np.random.normal(cell_stats[1][0], cell_stats[1][1], 1)

        if(obj_s['feature'].lower() == "benign"):
            if(i<len(hardcoded_benign_bounding_boxes)):
                sampled_width = hardcoded_benign_bounding_boxes[i][0]
                sampled_height = hardcoded_benign_bounding_boxes[i][1]
            else:
                sampled_width = cell_stats[0][0]
                sampled_height = cell_stats[1][0]
        elif(obj_s['feature'].lower() == "malignant"):
            if (i < len(hardcoded_malignant_bounding_boxes)):
                sampled_width = hardcoded_malignant_bounding_boxes[i][0]
                sampled_height = hardcoded_malignant_bounding_boxes[i][1]
            else:
                sampled_width = cell_stats[0][0]
                sampled_height = cell_stats[1][0]
        else:
            sampled_width = cell_stats[0][0]
            sampled_height = cell_stats[1][0]

        x0 = sx0
        y0 = sy0
        if(x0 > 1):
            x0 = x0/image_size
        if (y0 > 1):
            y0 = y0/image_size

        x1 = x0 + sampled_width
        y1 = y0 + sampled_height
        bbox = torch.FloatTensor([x0, y0, x1, y1])

        objects_boxes_gt.append(torch.FloatTensor(bbox))

        v_int = vertex(point, object_name, objects_boxes_gt, doFloat=False)
        vertex_dict_int[v_int.object_id] = v_int
        vertices_int.append(v_int)

        object_id_to_index[v_int.object_id] = index
        index+=1

        # Create object embeddings and append it to list
        object_embeddings.append(create_embedding(object_name, v_int.point, bbox))

    img_sample_int = np.full((image_size,image_size,3), 255)
    img_sample_int = img_sample_int.astype(np.float32)
    img_sample_int = cv2.cvtColor(img_sample_int, cv2.COLOR_BGR2RGB)

    rect = (0, 0, image_size, image_size)
    subdiv_int = cv2.Subdiv2D(rect)

    ind = 0
    for p in points:
        subdiv_int.insert(p)
        draw_point(img_sample_int, p, color_dict[object_names[ind]])
        ind+=1

    delaunay_color = (128,128,128)

    edges_ = delaunay_triangulation(img_sample_int, subdiv_int, vertex_dict_int, delaunay_color=delaunay_color, draw=True, doFloat=False)
    img_sample_int =img_sample_int.astype(np.uint8)

    euclidean_distance_scaler = distance.euclidean((0, 0), image_size)
    for e in edges_:
        triples.append((object_id_to_index[e.v1.object_id], object_id_to_index[e.v2.object_id],
                        e.compute_edge_value()/euclidean_distance_scaler)) #divided by root 2 to match it with training settings

    object_embeddings = np.array(object_embeddings)
    object_embeddings = torch.FloatTensor(object_embeddings)
    triples = torch.FloatTensor(triples)
    objects_boxes_gt = torch.stack(objects_boxes_gt, dim=0)

    return object_embeddings, triples, objects_boxes_gt, img_sample_int
