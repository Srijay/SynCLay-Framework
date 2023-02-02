import os
import inspect
import time
import torch
import subprocess
from contextlib import contextmanager
import imageio as io
from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt

import random
import shutil
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.autograd import Variable

def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def remove_alpha_channel(img):
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img


def display_image(img):
    plt.imshow(img)
    plt.show()


def save_numpy_image_INT(img,path):
    im = Image.fromarray(img)
    im.save(path)


def save_numpy_image_FLOAT(img,path):
    matplotlib.image.imsave(path, img)


def save_numpy_image_imageio(img,path):
    io.imwrite(path,img)


def int_tuple(s):
  return tuple(int(i) for i in s.split(','))


def float_tuple(s):
  return tuple(float(i) for i in s.split(','))


def str_tuple(s):
  return tuple(s.split(','))


@contextmanager
def timeit(msg, should_time=True):
  if should_time:
    torch.cuda.synchronize()
    t0 = time.time()
  yield
  if should_time:
    torch.cuda.synchronize()
    t1 = time.time()
    duration = (t1 - t0) * 1000.0
    print('%s: %.2f ms' % (msg, duration))


def bool_flag(s):
  if s == '1':
    return True
  elif s == '0':
    return False
  msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
  raise ValueError(msg % s)


def lineno():
  return inspect.currentframe().f_back.f_lineno


def get_gpu_memory():
  torch.cuda.synchronize()
  opts = [
      'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
  ]
  cmd = str.join(' ', opts)
  ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  output = ps.communicate()[0].decode('utf-8')
  output = output.split("\n")[1].split(":")
  consumed_mem = int(output[1].strip().split(" ")[0])
  return consumed_mem

class LossManager(object):
  def __init__(self):
    self.total_loss = None
    self.all_losses = {}

  def add_loss(self, loss, name, weight=1.0):
    cur_loss = loss * weight
    if self.total_loss is not None:
      self.total_loss += cur_loss
    else:
      self.total_loss = cur_loss

    self.all_losses[name] = cur_loss.data.cpu().item()

  def items(self):
    return self.all_losses.items()


