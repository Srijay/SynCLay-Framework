import argparse
import json
import math
import os
import sys
from datetime import datetime

import cv2
import matplotlib
import numpy as np
import torch
from torchvision.utils import save_image

sys.path.insert(0,'C:/Users/Srijay/Desktop/Projects/scene_graph_pathology')
#sys.path.insert(0,'D:/warwick/scene_graph_pathology')

from model import Sg2ImModel
from image_from_scene_graph import hovernet_class_output_to_class_image, colored_images_from_classes
from utils import mkdir, save_numpy_image_FLOAT

from preprocessing.delaunay_triangulation_glands import vertex, delaunay_triangulation, draw_point

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='./../outputs/conic_hovernet_integrated/model_1.pt')
parser.add_argument('--output_dir', default='outputs/gui')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

#Object vector parameters
parser.add_argument('--use_size_feature', default=0, type=int)
parser.add_argument('--use_loc_feature', default=1, type=int)

args = parser.parse_args()

VOCAB = {"neutrophil": 0, "epithelial": 1, "lymphocyte": 2, "plasma": 3, "eosinophil": 4, "connectivetissue": 5}


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
    dirname = os.path.dirname(args.checkpoint)
    model = Sg2ImModel(**checkpoint['model_kwargs'])
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)

    model.eval()
    model.to(device)
    return model


def json_to_img(scene_graph, model):

    object_embeddings, triples, objects_boxes_gt, img_sample = json_to_scene_graph(scene_graph)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # Run the model forward
    with torch.no_grad():

        if torch.cuda.is_available():
            object_embeddings = object_embeddings.cuda()
            triples = triples.cuda()
            objects_boxes_gt = objects_boxes_gt.cuda()

            image_pred, label_pred_hovernet_patches, object_boxes, object_bounding_boxes_preds, _ = model(object_indices=None,
                                                                                                            object_coordinates=None,
                                                                                                            object_embeddings=object_embeddings,
                                                                                                            triples=triples,
                                                                                                            objects_boxes_gt=objects_boxes_gt)

            label_pred_hovernet_class_output = np.argmax(label_pred_hovernet_patches.cpu().numpy(), axis=1)
            label_pred = hovernet_class_output_to_class_image(label_pred_hovernet_class_output)
            label_pred_img = colored_images_from_classes(label_pred, 256)

    image_output_dir = os.path.join(args.output_dir,'images')
    mask_output_dir = os.path.join(args.output_dir,'masks')
    mkdir(image_output_dir)
    mkdir(mask_output_dir)

    # Save the component mask layout and generated image
    mask_path = os.path.join(mask_output_dir, 'mask{}.png'.format(current_time))
    save_numpy_image_FLOAT(label_pred_img, mask_path)
    img_path = os.path.join(image_output_dir, 'image{}.png'.format(current_time))
    save_image(image_pred[0], img_path)

    return img_path, mask_path


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
        H, W = 256,256
        location_vector[0] = object_location[0] / W
        location_vector[1] = object_location[1] / H  # Scaling the locations
        embedding = embedding + location_vector

    if (args.use_size_feature):  # Size is determined by area of gland
        size_embedding = [0.0] * 10
        max_area = 256 * 256.0
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

    print(json_text)

    scene = json.loads(json_text)

    if len(scene) == 0:
        return []

    image_id = scene['image_id']
    scene = scene['objects']
    objects = [i['text'].lower() for i in scene]
    image_size = (256,256)

    object_embeddings = []
    triples = []
    objects_boxes_gt = []
    vertices = []
    vertices_int = []
    points = []
    object_id_to_index = {}
    index = 0

    for i in range(0, len(objects)):
        obj_s = scene[i]
        # build vertex info
        sx0 = obj_s['left']
        sy0 = obj_s['top']
        if(sx0==0):
            sx0=0.0
        point = (sx0,sy0)
        points.append(point)
        #get bounding box info
        sx1 = obj_s['width'] + sx0
        sy1 = obj_s['height'] + sy0
        margin = (obj_s['size'] + 1) / 10 / 2
        mean_x_s = 0.5 * (sx0 + sx1)
        mean_y_s = 0.5 * (sy0 + sy1)
        sx0 = max(0, mean_x_s - margin)
        sx1 = min(1, mean_x_s + margin)
        sy0 = max(0, mean_y_s - margin)
        sy1 = min(1, mean_y_s + margin)
        bbox = [sx0, sy0, sx1, sy1]
        objects_boxes_gt.append(torch.FloatTensor(bbox))
        object_name = obj_s['text'].lower()
        v = vertex(point, object_name, objects_boxes_gt)
        vertices.append(v)
        v_int = vertex((int(point[0]*256),int(point[1]*256)), object_name, objects_boxes_gt)
        vertices_int.append(v_int)
        object_id_to_index[v.object_id] = index
        index+=1
        # Create object embeddings and append it to list
        object_embeddings.append(create_embedding(object_name, v.point, bbox))

    img_sample = np.full((256,256,3), 0.0)
    img_sample_int = np.full((256,256,3), 255)
    img_sample_int = img_sample_int.astype(np.float32)
    img_sample_int = cv2.cvtColor(img_sample_int, cv2.COLOR_BGR2RGB)
    rect = (0, 0, image_size[1], image_size[0])
    subdiv = cv2.Subdiv2D(rect)
    subdiv_int = cv2.Subdiv2D(rect)

    def adjust_range(point_coord):
        if(point_coord<0):
            point_coord = 0.0
        if(point_coord>1):
            point_coord = 1.0
        return point_coord

    for p in points:
        p = (adjust_range(p[0]),adjust_range(p[1]))
        subdiv.insert(p)
        p1 = (int(p[0]*256),int(p[1]*256))
        subdiv_int.insert(p1)
        draw_point(img_sample_int, p1, (0, 0, 255))

    edges = delaunay_triangulation(img_sample, subdiv, vertices, draw=False, doFloat=True)
    delaunay_color = (169, 54, 169)
    edges_ = delaunay_triangulation(img_sample_int, subdiv_int, vertices_int, delaunay_color=delaunay_color, draw=True)
    img_sample_int =img_sample_int.astype(np.uint8)

    #euclidean_distance_scaler = distance.euclidean((0, 0), image_size)
    for e in edges:
        triples.append((object_id_to_index[e.v1.object_id], object_id_to_index[e.v2.object_id],
                        e.compute_edge_value()/math.sqrt(2))) #divided by root 2 to match it with training settings

    object_embeddings = np.array(object_embeddings)
    object_embeddings = torch.FloatTensor(object_embeddings)
    triples = torch.FloatTensor(triples)
    objects_boxes_gt = torch.stack(objects_boxes_gt, dim=0)

    return object_embeddings, triples, objects_boxes_gt, img_sample_int
