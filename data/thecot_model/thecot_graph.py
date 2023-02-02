import pandas
import numpy as np
import os
import torch
import cv2
from scipy.spatial import distance
from PIL import Image
from numpy import *
import pickle

color_dict = {
        "neutrophil": [0, 0, 0],  # neutrophil  : black
        "epithelial": [0, 255, 0],  # epithelial : green
        "lymphocyte": [255, 255, 0],  # lymphocyte : Yellow
        "plasma": [255, 0, 0],  # plasma : red
        "eosinophil": [0, 0, 255],  # eosinophil : Blue
        "connectivetissue": [255, 0, 255],  # connectivetissue : fuchsia
    }

class TheCOTGraph():


    def __init__(self, cellular_layout_folder=None, cells_size_distribution_file=None, image_size=256, use_loc_feature=True):

        self.matlab_to_python_cell_names_map = {1 : "epithelial", 2 : "neutrophil", 3 : "lymphocyte", 4 : "eosinophil", 5 : "plasma", 6 : "connectivetissue"}

        self.vocab = {"neutrophil": 0, "epithelial": 1, "lymphocyte": 2, "plasma": 3, "eosinophil": 4 , "connectivetissue": 5} #Original vocab, used everywhere

        self.image_size = image_size

        self.use_loc_feature = use_loc_feature

        self.cellular_layout_folder = cellular_layout_folder

        #It includes mean and std of widths and heights of nuclei
        if (cells_size_distribution_file is not None):
            self.cells_size_distributions = pickle.load(open(cells_size_distribution_file, 'rb'))


    def sample_cell_size(self,cell_id):
        cell_stats = self.cells_size_distributions[cell_id]
        # sampled_width = np.random.normal(cell_stats[0][0], cell_stats[0][1]/8, 1)
        # sampled_height = np.random.normal(cell_stats[1][0], cell_stats[1][1]/8, 1)
        return cell_stats[0][0], cell_stats[1][0]
        return sampled_width, sampled_height


    def create_embedding(self, cell_id, cell_location):

        embedding = []

        class_vector = [0.0]*len(self.vocab)
        class_vector[cell_id] = 1.0
        embedding = embedding+class_vector

        if (self.use_loc_feature):
            location_vector = [0.0]*2 #vector of size 2 for location
            H = W = self.image_size
            location_vector[0] = cell_location[0]/W
            location_vector[1] = cell_location[1]/H #Scaling the locations
            embedding = embedding+location_vector

        return embedding


    def delaunay_triangulation(self, rect, subdiv, loc_to_cellindex_dict, delaunay_image=None, draw=False, draw_edges=True, delaunay_color=None):

        euclidean_distance_scaler = distance.euclidean((0, 0), self.image_size)

        # Check if a point is inside a rectangle
        def rect_contains(rect, point):
            if point[0] < rect[0]:
                return False
            elif point[1] < rect[1]:
                return False
            elif point[0] > rect[2]:
                return False
            elif point[1] > rect[3]:
                return False
            return True


        def add_edge_triplet(edge_triplets, edge_dict, cell1, cell2, loc1, loc2):
            edge_key = str(cell1) + "_" + str(cell2)
            if (edge_key not in edge_dict):
                edge_dict[edge_key] = 1
                edge_val = distance.euclidean(loc1, loc2)/euclidean_distance_scaler
                edge_triplets.append([cell1,cell2,edge_val])


        triangleList = subdiv.getTriangleList()

        edge_triplets = []
        edge_dict = {}

        for t in triangleList:

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            cell1_index = loc_to_cellindex_dict[str(int(t[0])) + "_" + str(int(t[1]))]
            cell2_index = loc_to_cellindex_dict[str(int(t[2])) + "_" + str(int(t[3]))]
            cell3_index = loc_to_cellindex_dict[str(int(t[4])) + "_" + str(int(t[5]))]

            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):

                if (draw):
                    if(draw_edges):
                        cv2.line(delaunay_image, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                        cv2.line(delaunay_image, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                        cv2.line(delaunay_image, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

                add_edge_triplet(edge_triplets, edge_dict, cell1_index, cell2_index, pt1, pt2)
                add_edge_triplet(edge_triplets, edge_dict, cell2_index, cell1_index, pt2, pt1)
                add_edge_triplet(edge_triplets, edge_dict, cell2_index, cell3_index, pt2, pt3)
                add_edge_triplet(edge_triplets, edge_dict, cell3_index, cell2_index, pt3, pt2)
                add_edge_triplet(edge_triplets, edge_dict, cell1_index, cell3_index, pt1, pt3)
                add_edge_triplet(edge_triplets, edge_dict, cell3_index, cell1_index, pt3, pt1)

        return edge_triplets, delaunay_image


    def sample_graph(self, draw=False, draw_edges=True, matlab_cellular_layout_file="", shrinked_cellular_layout_file="", output_path=""):

        filename, file_extension = os.path.splitext(matlab_cellular_layout_file)
        if (file_extension == ".txt"):
            self.cellular_layout = np.loadtxt(matlab_cellular_layout_file, delimiter=',', dtype=int32);
        else:
            self.cellular_layout = np.load(matlab_cellular_layout_file)

        count_dict = {'epithelial':0,
                      'neutrophil': 0,
                      'lymphocyte': 0,
                      'plasma': 0,
                      'eosinophil': 0,
                      'connectivetissue': 0}

        for id in self.matlab_to_python_cell_names_map:
            count_dict[self.matlab_to_python_cell_names_map[id]] = np.count_nonzero(self.cellular_layout==id)

        if self.cellular_layout.shape[0] != self.image_size:
            shrinking_factor = self.image_size / self.cellular_layout.shape[0]
            cellular_layout_new = np.zeros((self.image_size, self.image_size))
            for matlab_id in self.matlab_to_python_cell_names_map:
                i, j = np.where(self.cellular_layout == matlab_id)
                for k in range(0, len(i)):
                    cellular_layout_new[int(i[k] * shrinking_factor)][int(j[k] * shrinking_factor)] = matlab_id
            self.cellular_layout = cellular_layout_new

        #save the shrinked cellular layout
        np.save(shrinked_cellular_layout_file, self.cellular_layout)

        # Draw a point
        def draw_point(img, p, color):
            cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)

        cell_ids = []
        cell_locations = []
        cell_names = []
        for matlab_id in self.matlab_to_python_cell_names_map:
            i, j = np.where(self.cellular_layout == matlab_id)
            location = list(zip(i, j))
            cell_ids+=[self.vocab[self.matlab_to_python_cell_names_map[matlab_id]]]*len(location)
            cell_locations+=location
            cell_names+=[self.matlab_to_python_cell_names_map[matlab_id]]*len(location)

        #For delaunay triangulation
        rect = (0, 0, self.image_size, self.image_size)
        subdiv = cv2.Subdiv2D(rect)

        delaunay_image=None
        if(draw):
            delaunay_image = np.full((self.image_size,self.image_size,3), 255)

        cell_embeddings = []
        cell_bounding_boxes = []
        cell_id_to_index = {}
        loc_to_cellindex_dict = {}

        for i in range(0,len(cell_ids)):

            cell_id_to_index[cell_ids[0]] = i

            cell_embeddings.append(self.create_embedding(cell_ids[i],cell_locations[i]))

            w,h = self.sample_cell_size(cell_ids[i])

            # Extract bounding boxes and append
            x0 = cell_locations[i][0] / self.image_size
            y0 = cell_locations[i][1] / self.image_size
            x1 = x0 + w
            y1 = y0 + h
            cell_bounding_boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

            # For Delanay Triangulation
            loc_to_cellindex_dict[str(cell_locations[i][0]) + "_" + str(cell_locations[i][1])] = i
            subdiv.insert((cell_locations[i][0],cell_locations[i][1]))

            if(draw):
                draw_point(delaunay_image, (cell_locations[i][0],cell_locations[i][1]), color_dict[cell_names[i]])

        edge_triplets, delaunay_image = self.delaunay_triangulation(rect, subdiv, loc_to_cellindex_dict, delaunay_image=delaunay_image, draw=draw, draw_edges=draw_edges, delaunay_color=(128,128,128))

        object_embeddings = torch.FloatTensor(cell_embeddings)
        object_bounding_boxes = torch.stack(cell_bounding_boxes, dim=0)
        edge_triplets = torch.FloatTensor(edge_triplets)

        if(draw):
            delaunay_image = delaunay_image.astype(np.uint8)
            Image.fromarray(delaunay_image).save(output_path)

        return object_embeddings, object_bounding_boxes, edge_triplets, count_dict


if __name__ == '__main__':

    thecot_graph = TheCOTGraph(cellular_layout_file="cellular_layouts/1.txt",
                               cells_size_distribution_file="cells_size_distributions.obj",
                               image_size=256)

    object_embeddings, object_bounding_boxes, edge_triplets = thecot_graph.sample_graph(draw=True,
                                                                                        output_path="./sample.png")