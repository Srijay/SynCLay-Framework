# example of calculating the frechet inception distance in Keras
import numpy
import glob
import random
import os
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    #act1 = numpy.concatenate((act1,act1),axis=0)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

folder_1 = "C:/Users/Srijay/Desktop/Projects/scene_graph_pathology/outputs/conic_hovernet_integrated/test/model2/gt_image"
folder_2 = "C:/Users/Srijay/Desktop/Projects/scene_graph_pathology/outputs/conic_hovernet_integrated/test/model2/pred_image"
scale = 0
size = 256
max_file_num = 200

paths1 = [file for file in os.listdir(folder_1) if file.endswith('.png')]
paths2 = [file for file in os.listdir(folder_2) if file.endswith('.png')]

paths1 = random.sample(paths1,max_file_num)
paths2 = random.sample(paths2,max_file_num)

length = len(paths1)

print("Number of files to be processed: ",length)

def get_images(folder,file_names):
    images = []
    for fname in file_names:
        img = Image.open(os.path.join(folder,fname))
        img = numpy.asarray(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        images.append(img)
    return images

images_random = randint(0, 255, length*size*size*3)
images_random = images_random.reshape((length,size,size,3))
images_random = images_random.astype('float32')

images_real = get_images(folder_1,paths1)
images_gen = get_images(folder_2,paths2)

images_real = numpy.array(images_real)
images_gen = numpy.array(images_gen)

# prepare the inception v3 model
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(size,size,3))

# pre-process images
images_real = preprocess_input(images_real)
images_gen = preprocess_input(images_gen)
images_random = preprocess_input(images_random)

fid = calculate_fid(model, images_real, images_real)
print('FID (same): %.3f' % fid)

fid = calculate_fid(model, images_real, images_random)
print('FID (random): %.3f' % fid)

fid = calculate_fid(model, images_real, images_gen)
print('FID (predicted) : %.3f' % fid)