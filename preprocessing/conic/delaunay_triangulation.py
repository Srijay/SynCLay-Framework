import matplotlib
import matplotlib.pyplot as plt
import imutils
import glob
import os, sys
from scipy.spatial import distance
import numpy as np
import shutil
import copy
from PIL import Image
import cv2

sys.path.insert(0, os.getcwd())

from utils import remove_alpha_channel


# Vertex of a graph
class vertex:
    def __init__(self, point, object_name, bounding_box):
        self.point = point
        #self.object_id = str(round(point[0], 2)) + "_" + str(round(point[1], 2))
        self.object_id = str(int(point[0])) + "_" + str(int(point[1]))
        self.object_name = object_name
        self.bounding_box = bounding_box

    def print_vertex_info(self):
        print("Object Name: ", self.object_name)
        print("Object Coordinates ", self.point)
        print("Object Bounding Box: ", self.bounding_box)


# Graph is defined by edges
class edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def compute_edge_value(self):
        return distance.euclidean(self.v1.point, self.v2.point)

    def print_edge_info(self):
        print("Starting vertex: ", self.v1.object_id)
        print("Ending vertex: ", self.v2.object_id)
        print("Distance between them: ", self.compute_edge_value())


# Draw a point
def draw_point(img, p, color):
    label_color = (0, 0, 0)
    cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)
    # if(p[0]==159):
    #cv2.putText(img, str(p[0]) + ',' + str(p[1]), (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, label_color, 2)


# Draw a rectangle
def draw_rectangle(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)


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


def retrieve_vertex(vertices, pt_id):
    for v in vertices:
        if (v.object_id == pt_id):
            return v
    raise "Error in retrieve_vertex function in delaunay_triangulation.py"


# Draw delaunay triangles
# Object O is in form of [object_id, x1, y1 (centroid location)]
# Triplet: (O1,O2,distance)
def delaunay_triangulation(img, subdiv, vertex_dict, delaunay_color=None, draw=False, doFloat=False):

    def add_edge(edges,edge_dict,vertex_1,vertex_2):
        edge_key_1_2 = vertex_1.object_id + "_" + vertex_2.object_id
        if (edge_key_1_2 not in edge_dict):
            edge_pt1_pt2 = edge(vertex_pt1, vertex_pt2)
            edge_dict[edge_key_1_2] = 1
            edges.append(edge_pt1_pt2)

    triangleList = subdiv.getTriangleList()

    size = img.shape
    r = (0, 0, size[1], size[0])
    edges = []
    edge_dict = {}


    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):

            if (draw):
                cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

            if (doFloat):
                vertex_pt1 = retrieve_vertex(vertices, str(round(t[0], 2)) + "_" + str(round(t[1], 2)))
                vertex_pt2 = retrieve_vertex(vertices, str(round(t[2], 2)) + "_" + str(round(t[3], 2)))
                vertex_pt3 = retrieve_vertex(vertices, str(round(t[4], 2)) + "_" + str(round(t[5], 2)))
            else:

                key_0_1 = str(int(t[0])) + "_" + str(int(t[1]))
                vertex_pt1 = vertex_dict[key_0_1]

                key_2_3 = str(int(t[2])) + "_" + str(int(t[3]))
                vertex_pt2 = vertex_dict[key_2_3]

                key_4_5 = str(int(t[4])) + "_" + str(int(t[5]))
                vertex_pt3 = vertex_dict[key_4_5]

                add_edge(edges,edge_dict,vertex_pt1,vertex_pt2)
                add_edge(edges,edge_dict,vertex_pt2,vertex_pt1)
                add_edge(edges,edge_dict,vertex_pt2,vertex_pt3)
                add_edge(edges,edge_dict,vertex_pt3,vertex_pt2)
                add_edge(edges,edge_dict,vertex_pt1,vertex_pt3)
                add_edge(edges,edge_dict,vertex_pt3,vertex_pt1)

    return edges



def get_nuclei_type(color_mask, point, bounding_box):
    check_point = color_mask[point[1]][point[0]].copy()

    if (np.array_equal(check_point, [255, 255, 255])):
        x, y, w, h = bounding_box
        check_point[0] = int(np.mean(color_mask[y:y + h, x:x + h, 0]))
        check_point[1] = int(np.mean(color_mask[y:y + h, x:x + h, 1]))
        check_point[2] = int(np.mean(color_mask[y:y + h, x:x + h, 2]))
        max_elem = max(check_point[0],check_point[1],check_point[2])
        check_point[check_point<max_elem] = 0
        check_point[check_point>=max_elem] = 255

    if (np.array_equal(check_point, [0,0,0])):
        return "neutrophil"
    if (np.array_equal(check_point, [0,255,0])):
        return "epithelial"
    if (np.array_equal(check_point, [255,255,0])):
        return "lymphocyte"
    if (np.array_equal(check_point, [255,0,0])):
        return "plasma"
    if (np.array_equal(check_point, [0,0,255])):
        return "eosinophil"
    if (np.array_equal(check_point, [255,0,255])):
        return "connectivetissue"

    print(point)
    print(color_mask[point[1]][point[0]])
    raise Exception('Nuclei Type Not Detected')


# Create delaunay graph and store it
def delaunay_scene_graph(input_mask_path, img_size, draw=False, output_path=None):

    # Read color mask and do pre-processing
    color_mask = Image.open(input_mask_path).resize(img_size)
    color_mask = remove_alpha_channel(np.asarray(color_mask))

    mask = np.asarray(color_mask).copy()
    tot = mask.sum(-1)
    mask[:, :, 0][tot[:, :]>=550] = 0
    mask[:, :, 1][tot[:, :]>=550] = 0
    mask[:, :, 2][tot[:, :]>=550] = 0
    mask[:, :, 0][tot[:, :]<550] = 255
    mask[:, :, 1][tot[:, :]<550] = 255
    mask[:, :, 2][tot[:, :]<550] = 255

    gray_mask = np.asarray(Image.fromarray(mask).convert('L')).copy()
    gray_mask[gray_mask > 128] = 255
    gray_mask[gray_mask <= 128] = 0

    # Image.fromarray(color_mask).save("color_mask.png")
    # Image.fromarray(gray_mask).save("gray_mask.png")
    # Image.fromarray(mask).save("mask.png")

    # Computation of centroids of blobs
    cnts = cv2.findContours(gray_mask.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    points = []
    vertices = []
    vertex_dict = {}
    bounding_boxes = []
    object_names = []


    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)

        if(M["m00"]==0):
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        point = (cX, cY)
        points.append((cX, cY))

        # Compute bounding box
        x, y, w, h = cv2.boundingRect(c)
        bounding_box = (x, y, w, h)
        bounding_boxes.append(bounding_box)

        #Retrieve the nuclei type
        object_name = get_nuclei_type(color_mask,point,bounding_box)

        new_vertex = vertex(point, object_name, bounding_box)
        vertex_dict[new_vertex.object_id] = new_vertex
        vertices.append(new_vertex)

        object_names.append(object_name)

    # Computation of delaunay triangle
    size = color_mask.shape
    rect = (0, 0, size[1], size[0])

    # Define window names
    win_delaunay = "Delaunay Triangulation"

    # Define colors for drawing.
    delaunay_color = (255, 0, 255)

    subdiv = cv2.Subdiv2D(rect)
    img_copy = color_mask.copy()

    # Turn on animation while drawing triangles
    # Insert points into subdiv

    for p in points:
        subdiv.insert(p)
        # Draw delaunay triangles
        # if draw:
        #     # Show animation
        #     delaunay_triangulation(img_copy, subdiv, vertices, delaunay_color, draw=True)
            # cv2.imshow(win_delaunay, img_copy)
            # cv2.waitKey(100)

    if draw:
        # Draw points
        for p in points:
            draw_point(img_copy, p, (0, 0, 255))

        # for bbx in bounding_boxes:
        #     draw_rectangle(img_copy, bbx[0], bbx[1], bbx[2], bbx[3])

        delaunay_triangulation(img_copy, subdiv, vertex_dict, delaunay_color, draw=True)

        #Show results
        # cv2.imshow(win_delaunay, img_copy)
        # cv2.waitKey(500)

    # Draw delaunay triangles
    edges = delaunay_triangulation(img_copy, subdiv, vertex_dict)

    if (output_path):
        Image.fromarray(img_copy).save(output_path)

    del color_mask
    del mask
    del img_copy
    del cnts
    del subdiv

    return gray_mask, points, bounding_boxes, vertices, edges


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


