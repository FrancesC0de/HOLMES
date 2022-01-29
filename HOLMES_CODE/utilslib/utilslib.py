# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile

import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def imshow(image, title=None, ret=True):
    image = image.cpu().numpy()
    image = np.transpose(image, (1,2,0)) 
    mean = np.array(list(MEAN))
    std = np.array(list(STD))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    if ret == True:
      return image
    plt.imshow(image)
    if title is not None:
      plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def imshow_t(inp, title=None, ret=False):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(list(MEAN))
    std = np.array(list(STD))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if ret == True:
      return inp
    plt.imshow(inp)
    if title is not None:
      plt.title(title)
    plt.pause(0.001)
    
def imshowtemp(inp, title=None, ret = False):
    cv2.imwrite('tmp.jpg', inp) 
    image = cv2.imread('tmp.jpg')
    if os.path.exists('tmp.jpg'):
      os.remove('tmp.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if ret == True:
      return image
    plt.figure(figsize=(20,4))
    plt.axis('off')
    if title is not None:
      plt.title(title)
    plt.imshow(image)
    plt.show()
    
def interpolate_img_hm(img, heatmap, img_r = 0.6, hm_r = 0.4, size = (224, 224), color = False):
    if color == True:
      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * hm_r + img_r * img
    superimposed_img = cv2.resize(superimposed_img, size) 
    return superimposed_img

def process_image(image_path, preprocess=True, resize = False):
    image = Image.open(image_path)
    if image.mode != 'RGB':
      image = image.convert('RGBA')
    image = image.convert('RGB')
    
    if resize == False:
      img = image.resize((224, 224))
    else:
      # Resize
      img = image.resize((256, 256))

      # Center crop
      width = 256
      height = 256
      new_width = 224
      new_height = 224

      left = (width - new_width) / 2
      top = (height - new_height) / 2
      right = (width + new_width) / 2
      bottom = (height + new_height) / 2
      img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1))
    if preprocess == True:
      img = img /256

      # Standardization
      means = np.array(list(MEAN)).reshape((3, 1, 1))
      stds = np.array(list(STD)).reshape((3, 1, 1))

      img = img - means
      img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor
    
# Given label number returns class name
def get_class_name(c):
    try:
        labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    except:
        labels = np.loadtxt('../synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])
    
def plot_row(images, columns, titles = None):
  fig=plt.figure(figsize=(20, 20))
  rows = 1
  for i in range(1, columns*rows +1):
      img = images[i-1]
      ax = fig.add_subplot(rows, columns, i)
      if titles is not None:
        ax.set_title(titles[i-1])
      plt.imshow(img)
      plt.axis('off')
  plt.show()

def loop_inplace_sum(arrlist):
    # assumes len(arrlist) > 0
    sum = arrlist[0].copy()
    for a in arrlist[1:]:
        sum += a
    return sum