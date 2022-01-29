# -*- coding: utf-8 -*-

import functools
import multiprocessing as mp
import os
import re

from keras.preprocessing import image
from PIL import Image

from tqdm import tqdm

from collections import OrderedDict

import numpy as np
from sklearn.decomposition import PCA

import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

import pandas as pd

import matplotlib.pyplot as plt
import warnings

"""**Get network activations**"""

def get_files(imagedir, ext='jpg|jpeg|JPG|JPEG|bmp|png|webp|PNG|BMP|WEBP'):
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(imagedir,base) for base in os.listdir(imagedir)
            if rex.match(base)]

def _image_worker(filename, size):
    try:
        img = Image.open(filename)
        if img.mode != 'RGB':
            img = img.convert('RGBA')
        img = img.convert('RGB').resize(size, resample=3)
        arr = image.img_to_array(img, dtype=int)
        return filename, arr
    except OSError as ex:
        print(f"skipping {filename}: {ex}")
        return filename, None

def read_images(imagedir, size):    
    ret = {}
    filenames = get_files(imagedir)
    for filename in tqdm(filenames):
        filename, arr = _image_worker(filename, size)
        ret[filename] = arr
    # {filename: 3d array (height, width, 3), ...}
    return {k: v for k,v in ret.items() if v is not None}

def get_model(layer='fc2'):
    with tf.device('/cpu:0'):
        base_model = VGG16(weights='imagenet', include_top=True)
        model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer(layer).output)
    return model

def fingerprint(image, model):
    if image.shape[2] == 1:
        image = image.repeat(3, axis=2)

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(image, axis=0)

    arr4d_pp = preprocess_input(arr4d)
    with tf.device('/cpu:0'):
        return model.predict(arr4d_pp)[0,:]

def get_activations(images, model):
    fingerprints = {}
    for fn, image in tqdm(images.items()):
        fingerprints[fn] = fingerprint(image, model)
    return fingerprints

def pca(fingerprints, n_components=0.9):
    fingerprints = OrderedDict(fingerprints)
    X = np.array(list(fingerprints.values()))
    Xp = PCA(n_components=n_components).fit(X).transform(X)
    return {k:v for k,v in zip(fingerprints.keys(), Xp)}

def get_data(image_dir, pca_apply=None):
  images = read_images(image_dir, size=(224,224))
  activations = get_activations(images, get_model())
  if pca_apply is not None:
    activations = pca(activations)
  filenames = list(activations.keys())
  X = np.array(list(activations.values()))
  return X, filenames

def show_outliers_prob(filenames, test_probs, t=0.5):
  outliers = []
  inliers = []
  for filename, prob in zip(filenames, test_probs):
    if prob > t:
      outliers.append(filename)
    else:
      inliers.append(filename)

  print('# Outliers: {}, # Retained: {}/{}'.format(len(outliers), len(inliers), len(filenames)))

  return inliers

"""**Outlier Detection**"""

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

from pyodds.algo.pca import PCA

def get_anomaly_score(X_df, contamination = 0.15, random_state = 42):
  # use Principal Component Classifier by default
  clf = PCA(random_state=random_state, contamination=contamination)
  # train the anomaly detection algorithm
  clf.fit(X_df)

  # get outlier result and scores
  prediction_result = clf.predict(X_df)
  outlierness_score = clf.decision_function(X_df)

  return outlierness_score, clf

def anomaly_likelihood(outlier_score, threshold):
    if threshold is not None:
      diff = outlier_score - threshold
      mask = diff > 0

      sc_pos = diff.clip(min=0)
      sc_neg = diff.clip(max=0)

      lmn = np.copy(diff)
      sc_pos = np.interp(sc_pos, (sc_pos.min(), sc_pos.max()), (0.5, 1))
      sc_neg = np.interp(sc_neg, (sc_neg.min(), sc_neg.max()), (0.0, 0.5))

      lmn[mask] = sc_pos[mask]
      lmn[np.logical_not(mask)] = sc_neg[np.logical_not(mask)]
      del diff, sc_pos, sc_neg
    else:
      mask = outlier_score < 0

      sc_pos = outlier_score.clip(max=0)
      sc_neg = outlier_score.clip(min=0)

      lmn = np.copy(outlier_score)
      sc_pos = np.interp(sc_pos, (sc_pos.min(), sc_pos.max()), (1, 0.5))
      sc_neg = np.interp(sc_neg, (sc_neg.min(), sc_neg.max()), (0.5, 0.0))

      lmn[mask] = sc_pos[mask]
      lmn[np.logical_not(mask)] = sc_neg[np.logical_not(mask)]
      del sc_pos, sc_neg
    return lmn

def show_anomaly_scores(X_df, filenames, contamination, threshold=None):
  outlierness_score, clf = get_anomaly_score(X_df, contamination)
  if hasattr(clf, 'threshold'):
    threshold = clf.threshold
  normalized_anomaly_scores = anomaly_likelihood(outlierness_score, threshold)
  inliers = show_outliers_prob(filenames, normalized_anomaly_scores)
  return inliers

def remove_outliers(image_dir, contamination=0.15):
  # retrieve the activations for the input samples and the input filenames
  X, filenames = get_data(image_dir)
  X_df = pd.DataFrame(X, index=filenames)
  # get the inliers' filenames and filter out the outliers
  inliers = show_anomaly_scores(X_df, filenames, contamination)
  return inliers
