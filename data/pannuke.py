import glob
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from scipy.spatial import distance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0,os.getcwd())

from preprocessing.pannuke.delaunay_triangulation import delaunay_scene_graph
from utils import remove_alpha_channel,save_numpy_image_INT
from torchvision.utils import save_image


class PanNukeDataset(Dataset):


    def __init__(self, image_dir, mask_dir, image_size=(256, 256), object_mask_size=(32,32),
                 min_object_size=0.02, min_objects_per_image=3, max_objects_per_image=8, use_size_feature=0, use_loc_feature=0):
        """
        A PyTorch Dataset for loading PanNuke images along with their masks, and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - mask_dir: Path to a directory where images are held
        - image_size: Size (H, W) at which to load images. Default (256, 256).
        - mask_size: Size M for object segmentation masks; default 32.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        """
        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.object_mask_size = object_mask_size
        self.use_size_feature = use_size_feature
        self.size_feature_num = 10
        self.use_loc_feature = use_loc_feature
        self.loc_feature_num = 2
        self.image_paths = glob.glob(os.path.join(self.image_dir,"*.png"))
        self.image_names = os.listdir(self.image_dir)
        self.mask_paths = glob.glob(os.path.join(self.mask_dir, "*.png"))
        self.mask_names = os.listdir(self.mask_dir)

        self.vocab = {"Neoplastic": 0, "Inflammatory": 1, "Soft": 2, "Dead": 3, "Epithelial": 4}
        self.embedding_dim = len(self.vocab) #one hot encoding

        if (self.use_loc_feature):
            self.embedding_dim += self.loc_feature_num #length of location feature (x,y) coordinates

        if (self.use_size_feature):
            self.embedding_dim += self.size_feature_num #length of size feature embedding


    def read_image(self,img_path):
        img = Image.open(img_path)
        img = img.resize(self.image_size)
        return remove_alpha_channel(np.asarray(img))


    def determine_nuclei(self,x):
        if (x==0):
            return "Neoplastic"
        if (x == 1):
            return "Inflammatory"
        if (x == 2):
            return "Soft"
        if (x == 3):
            return "Dead"
        if (x == 4):
            return "Epithelial"
        print("Error in determining grade")
        exit(0)


    def create_class_one_hot_vector(self, object_name):

        return self.vocab[object_name]


    def create_embedding(self,object_name,object_location,coarse_bbox):

        embedding = []

        class_vector = [0.0]*len(self.vocab)
        class_vector[self.vocab[object_name]] = 1.0
        embedding = embedding+class_vector

        if (self.use_loc_feature):
            location_vector = [0.0]*self.loc_feature_num #vector of size 2 for location
            H, W = self.image_size
            location_vector[0] = object_location[0]/W
            location_vector[1] = object_location[1]/H #Scaling the locations
            embedding = embedding+location_vector

        if(self.use_size_feature): #Size is determined by area of gland
            size_embedding = [0.0]*self.size_feature_num
            max_area = 256*256.0
            xsize = coarse_bbox[1]-coarse_bbox[0]
            ysize = coarse_bbox[3]-coarse_bbox[2]
            object_bbox_area = xsize*ysize
            size_index = int(object_bbox_area*10.0/max_area)
            if(size_index>9):
                size_index=9
            size_embedding[size_index] = 1
            embedding = embedding+size_embedding

        return embedding


    def get_constructed_bounding_box(self,p):

        x_v = p[0]
        y_v = p[1]
        x_img = self.image_size[0]
        y_img = self.image_size[1]

        x0 = x_v-self.object_mask_size[0]/2
        x1 = x_v+self.object_mask_size[1]/2
        y0 = y_v-self.object_mask_size[0]/2
        y1 = y_v+self.object_mask_size[1]/2

        if(x0<0):
            diff = -x0
            x0 = 0
            x1 = x1+diff
        if(x1>x_img):
            diff = x1-x_img
            x0 = x0-diff
            x1 = x_img
        if(y0<0):
            diff = -y0
            y0=0
            y1=y1+diff
        if(y1>y_img):
            diff = y1-y_img
            y0 = y0-diff
            y1 = y_img

        return x0,y0,x1,y1


    def __len__(self):
        return len(self.mask_paths)


    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_name = self.mask_names[index]
        image_path = os.path.join(self.image_dir,image_name)
        mask_path = os.path.join(self.mask_dir,image_name)

        color_mask = self.read_image(mask_path)
        # save_numpy_image_INT(color_mask,"investigate/color_mask_gt.png")

        image = self.read_image(image_path)

        try:
            mask, points, bounding_boxes, object_vertices, edges = delaunay_scene_graph(input_mask_path=mask_path,
                                                                                        img_size=self.image_size,
                                                                                        draw=True,
                                                                                        output_path="./../investigate/delanay_graph.png")

        except:
            print("Error")
            return

        exit()
        if len(object_vertices)==0 or np.mean(color_mask)<0.0001 or np.mean(color_mask)==255: #special case when image is empty
            return

        # for v in object_vertices:
        #     v.print_vertex_info()

        mask = mask/255.0
        color_mask = color_mask/255.0
        image = image/255.0

        mask_t = torch.from_numpy(mask)
        #color_mask_t = torch.from_numpy(color_mask) #if want to generate color mask instead of binary mask

        transform = T.Compose([T.ToTensor()])
        image_t = transform(image)
        color_mask_t = transform(color_mask)

        object_id_to_index = {}
        index = 0
        object_indices, object_bounding_boxes, object_masks, object_embeddings = [],[],[],[]
        object_coordinates = []
        object_bounding_boxes_constructed = []
        class_vectors = []

        H, W = self.image_size

        for v in object_vertices:
            #Assign index to an object
            object_id_to_index[v.object_id] = index
            object_indices.append(index)
            index+=1

            #Extract bounding boxes and append
            x, y, w, h = v.bounding_box
            x0 = x / W
            y0 = y / H
            x1 = (x + w) / W
            y1 = (y + h) / H #Scaling the bounding boxes
            object_bounding_boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

            object_coordinates.append(torch.FloatTensor([v.point[0],v.point[1]]))

            # Crop the mask according to the bounding box, being careful to
            # ensure that we don't crop a zero-area region
            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            #crop mask
            object_mask = mask[my0:my1, mx0:mx1]
            object_mask = np.asarray(Image.fromarray(object_mask).resize(self.object_mask_size)).copy()
            object_mask[object_mask > 0.5] = 1.0
            object_mask[object_mask <= 0.5] = 0
            # plt.imsave('mask.png', mask, cmap=cm.gray)
            # plt.imsave('image.png',image)
            object_mask = torch.from_numpy(object_mask)
            object_masks.append(object_mask)

            # xsize = mx1-mx0
            # ysize = my1-my0
            # plt.imsave('object_mask_'+str(xsize)+'_'+str(ysize)+".png",object_mask, cmap=cm.gray)

            #Create object embeddings and append it to list
            object_embeddings.append(self.create_embedding(v.object_name,v.point,(mx0,mx1,my0,my1)))

            #create class vectors
            class_vectors.append(self.create_class_one_hot_vector(v.object_name))

            #constructed bounding box
            x0, y0, x1, y1 = self.get_constructed_bounding_box(v.point)
            x0 = x0 / W
            y0 = y0 / H
            x1 = x1 / W
            y1 = y1 / H  # Scaling the constructed bounding boxes
            object_bounding_boxes_constructed.append(torch.FloatTensor([x0, y0, x1, y1]))

        triples = []
        euclidean_distance_scaler = distance.euclidean((0,0),self.image_size)
        for e in edges:
            triples.append((object_id_to_index[e.v1.object_id],object_id_to_index[e.v2.object_id],e.compute_edge_value()/euclidean_distance_scaler))

        object_bounding_boxes = torch.stack(object_bounding_boxes, dim=0)
        object_bounding_boxes_constructed = torch.stack(object_bounding_boxes_constructed, dim=0)
        object_masks = torch.stack(object_masks, dim=0)
        #object_indices = torch.LongTensor(object_indices)
        #object_embeddings = torch.stack(object_embeddings, dim=0)
        triples = torch.FloatTensor(triples)
        object_embeddings = np.array(object_embeddings)
        object_indices = np.array(object_indices)
        class_vectors = torch.LongTensor(class_vectors)

        return image_t, color_mask_t, object_indices, object_coordinates, object_bounding_boxes, object_bounding_boxes_constructed, object_masks, object_embeddings, triples, class_vectors


def pannuke_collate_fn(batch):
  """
  Collate function to be used when wrapping CocoSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving object categories
  - boxes: FloatTensor of shape (O, 4)
  - masks: FloatTensor of shape (O, M, M)
  - triples: LongTensor of shape (T, 3) giving triples
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - triple_to_img: LongTensor of shape (T,) mapping triples to images
  """

  if(batch[0]==None):
      return [[]]*10

  image_t_l = []
  mask_t_l = []
  for i, (image_t, mask_t, object_indices, object_coordinates, object_bounding_boxes, object_bounding_boxes_constructed, object_masks, object_embeddings, triples, class_vectors) in enumerate(batch):
    image_t_l.append(image_t[None])
    mask_t_l.append(mask_t[None])
    object_indices_l = object_indices
    object_bounding_boxes_l = object_bounding_boxes
    object_bounding_boxes_constructed_l = object_bounding_boxes_constructed
    object_masks_l = object_masks
    object_embeddings_l = torch.FloatTensor(object_embeddings)
    triples_l = triples
    object_coordinates_l = object_coordinates
    class_vectors_l = class_vectors

  image_t_l = torch.cat(image_t_l)
  mask_t_l = torch.cat(mask_t_l)
  out = (image_t_l, mask_t_l, object_indices_l, object_coordinates_l, object_bounding_boxes_l, object_bounding_boxes_constructed_l, object_masks_l, object_embeddings_l, triples_l, class_vectors_l)

  return out
