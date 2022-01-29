# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import imshow as cv2_imshow
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
import warnings

"""**Set working directory**"""

def set_wdir(S_ROOT, CKP_PATH, MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))

"""**Generate the explanations**"""

def generate_explanations(model_name, method, num_classes, test_images_path, meronyms, idx_to_class, checkpoint, printout=False, num_exp=None, masks_path='./explib/masks.npy'):
  explanations = []
  test_imgs_paths = os.listdir(test_images_path)
  # filter out folders
  test_imgs_paths = [x for x in test_imgs_paths if os.path.isfile(os.path.join(test_images_path, x))]
  sel_test_imgs_paths = test_imgs_paths if num_exp is None else test_imgs_paths[:np.min([num_exp, len(test_imgs_paths)])]
  for test_img_path in tqdm(sel_test_imgs_paths):
    # get the original model pretrained on ImageNet
    if model_name == 'vgg16':
      model = get_vgg_model() if (not method.startswith('gradcam')) else vgg16_ft(None, edit=False).cuda().eval()
    elif model_name == 'resnet50':
      model = get_resnet_model()
    elif model_name == 'densenet121':
      model = get_densenet_model()
    else:
      raise Exception('Model {} not supported.'.format(model_name))
    # generate the explanations according to the selected XAI method
    if method == 'holmes':
      drop_pred_values, meronym_explanations_dict = show_explanation(os.path.join(test_images_path, test_img_path), model, meronyms, num_classes, idx_to_class, checkpoint, printout=printout)
    elif method == 'gradcam':
      img_tensor, raw_heatmap, heatmap, class_idx = get_gradcam_results(model, os.path.join(test_images_path, test_img_path), class_idx=None, baseline=True, method='gradcam')
      drop_pred_values, meronym_explanations_dict = [(None, class_idx)], {'image': img_tensor, 'heatmap': raw_heatmap, 'heatmap_orig': heatmap}
    elif method == 'gradcam++':
      img_tensor, raw_heatmap, heatmap, class_idx = get_gradcam_results(model, os.path.join(test_images_path, test_img_path), class_idx=None, baseline=True, method='gradcam++')
      drop_pred_values, meronym_explanations_dict = [(None, class_idx)], {'image': img_tensor, 'heatmap': raw_heatmap, 'heatmap_orig': heatmap}
    elif method == 'rise':
      # create explainer instance 
      explainer = RISE(model, (224, 224))
      # generate the masks
      if not os.path.exists(masks_path): 
        # same hyper-parameters used in the official RISE code implementation
        explainer.generate_masks(N=5000, s=10, p1=0.1, savepath=masks_path) 
      else:
        explainer.load_masks(masks_path, p1=0.1)       
      img = process_image(os.path.join(test_images_path, test_img_path))
      img_tensor = Variable(img.cuda().unsqueeze(0)).cuda()
      heatmap, class_idx = explainer(img_tensor)
      raw_heatmap = cv2.resize(np.float32(heatmap), (224, 224)) 
      drop_pred_values, meronym_explanations_dict = [(None, class_idx)], {'image': img_tensor, 'heatmap': raw_heatmap} #, 'heatmap_orig': np.uint8(255 * heatmap)}
    else:
      raise Exception('XAI method {} not supported.'.format(method))
    explanations.append((test_img_path, meronym_explanations_dict, drop_pred_values))

  return explanations, len(sel_test_imgs_paths)
  
"""**Generate the insertion/deletion curves**"""

def generate_curves(model_name, method, MERONYMS, test_images_path, explanations, num_exp, verbose, n_rand_curves, klen=11, ksig=5, class_names_path='./explib/class_names.json'):
    # get heatmap and original image for curve metrics
    if model_name == 'vgg16':
      model = get_vgg_model()
    elif model_name == 'resnet50':
      model = get_resnet_model()
    elif model_name == 'densenet121':
      model = get_densenet_model()
    # Function that blurs input image
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2) 
    # Causal metrics
    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like) 
    # Curves dictionaries
    img_ins_curves = {}
    img_del_curves = {}
    img_rand_ins_curves = {}
    img_rand_del_curves = {}
    img_pred_classes = {}
    # generate the curves for each holonym image within the imagenet validation set
    for i in tqdm(range(num_exp)):
      img_path, meronym_explanations_dict, drop_pred_values = explanations[i]
      img_ins_curves[img_path] = []
      img_del_curves[img_path] = []
      img_rand_ins_curves[img_path] = []
      img_rand_del_curves[img_path] = []
      img_pred_classes[img_path] = []
      drop_values, pred_class = [v[0] for v in drop_pred_values], drop_pred_values[0][1]
      # XAI heatmap computation
      if method == 'holmes':
        final_heatmap, img_tensor, _, _, _, _ = create_global_heatmap(MERONYMS, meronym_explanations_dict, drop_values)
      else:
        img_tensor = meronym_explanations_dict['image']
        #heatmap_orig = meronym_explanations_dict['heatmap_orig'] 
        final_heatmap = meronym_explanations_dict['heatmap']
        # Load ImageNet class labels
        with open(class_names_path) as json_file: 
          class_names = json.load(json_file)        
        pred_class = class_names[str(pred_class)].split(',')[0]
      img_pred_classes[img_path].append(pred_class)
      
      # load the selected image
      path = os.path.join(test_images_path, img_path)
      img_orig = cv2.imread(path)
      img = cv2.resize(img_orig, (224, 224))     
      
      # segment the image
      with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        segments = np.array(slic(img_as_float(img)))
      # deletion
      h = deletion.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose)
      auc_deletion = auc(h)
      img_del_curves[img_path].append(auc_deletion)
      # insertion
      h = insertion.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose)
      auc_insertion = auc(h)
      img_ins_curves[img_path].append(auc_insertion)
      # random curves
      if n_rand_curves != 0:
        # random insertion(s)
        random_ins_aucs = []
        for i in range(n_rand_curves):
          h = insertion.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose, segments=segments.flatten())
          auc_insertion = auc(h)
          random_ins_aucs.append(auc_insertion)
        img_rand_ins_curves[img_path].append(random_ins_aucs)
        # random deletion(s)
        random_del_aucs = []
        for i in range(n_rand_curves):
          h = deletion.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose, segments=segments.flatten())
          auc_deletion = auc(h)
          random_del_aucs.append(auc_deletion)
        img_rand_del_curves[img_path].append(random_del_aucs)
        
    return img_del_curves, img_ins_curves, img_rand_del_curves, img_rand_ins_curves, img_pred_classes

  
"""**Evaluation**"""

def run_evaluation(CONFIG, HOL_MER_DICT, S_ROOT, CKP_PATH, TEST_IMGS_PATH, MODEL, METHOD, verbose=0, num_exp=50, n_rand_curves=5):
  # set random seed for reproducing the same random curves
  random.seed(42)
  # first check if the path to the holonyms folder (e.g., the ImageNet validation sets folder) is correct
  if os.path.exists(TEST_IMGS_PATH) == False:
    raise Exception('Holonyms folder not found. Cannot proceed without it.')
  # dictionary to serialize
  curves_dict = {}
  # load data if already existing
  curves_json = CONFIG + '_' + MODEL + '_' + METHOD + '_curves.json'
  if os.path.exists(curves_json) == True:
    with open(curves_json) as json_file: 
      curves_dict = json.load(json_file)

  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    CHECKPOINT = os.path.join(CKP_PATH, (HOLONYM + '.pth'))
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    MERONYMS_PATH = os.path.join(S_ROOT, HOLONYM)
    NUM_CLASSES = len(MERONYMS)
    IDX_TO_CLASS = { i : MERONYMS[i] for i in range(len(MERONYMS)) }
    # check if the path to the specific holonym test images exists
    try:
      test_images_path = os.path.join(TEST_IMGS_PATH, [x for x in os.listdir(TEST_IMGS_PATH) if (x.split(',')[0] == HOLONYM)][0])
      if os.path.exists(test_images_path) == False:
        raise Exception()
    except:
      raise Exception(('The ' + HOLONYM + ' holonym samples have not been found. Cannot generate their explanations without them.'))
    if HOLONYM not in curves_dict:
        # generate explanations according to the selected XAI method
        explanations, exp_qty = generate_explanations(MODEL, METHOD, NUM_CLASSES, test_images_path, MERONYMS, IDX_TO_CLASS, checkpoint=CHECKPOINT, num_exp=num_exp)
        # generate the curves for each holonym image according to the selected XAI method
        img_del_curves, img_ins_curves, img_rand_del_curves, img_rand_ins_curves, img_pred_classes = generate_curves(MODEL, METHOD, MERONYMS, test_images_path, explanations, exp_qty, verbose, n_rand_curves)
        # update the dictionary
        curves_dict[HOLONYM] = {}
        curves_dict[HOLONYM]['parts'] = MERONYMS
        curves_dict[HOLONYM]['del. curves'] = img_del_curves
        curves_dict[HOLONYM]['ins. curves'] = img_ins_curves
        if n_rand_curves != 0:
          curves_dict[HOLONYM]['random del. curves'] = img_rand_del_curves
          curves_dict[HOLONYM]['random ins. curves'] = img_rand_ins_curves
        curves_dict[HOLONYM]['pred_class'] = img_pred_classes
        # serialization
        with open(curves_json, 'w') as fp:
          json.dump(curves_dict, fp)

"""**Main**"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
                '--config',
                default='Scraping',
                choices=['Scraping', 'VOC'],
                help='load the configuration file for the scraping or the VOC setting')
  parser.add_argument(
                '--s_root',
                default='Image_scraping',
                help='scraping root folder where images are downloaded')
  parser.add_argument(
                '--ckp_path',
                default='Checkpoints',
                help='folder where model checkpoints are saved')
  parser.add_argument(
                '--test_imgs_path',
                default='Holonyms',
                help='folder where the test images for the holonyms are stored')
  parser.add_argument(
                '--root',
                default='../',
                help='root of the HOLMES main directory')
  parser.add_argument(
                '--main_dir',
                default='HOLMES_CODE',
                help='partial path from the root (e.g /My Drive/) to the HOLMES folder')
  parser.add_argument(
                '--model',
                default='vgg16',
                help='CNN model to be explained')
  parser.add_argument(
                '--method',
                default='holmes',
                help='Explanation method to be applied')
  parser.add_argument(
                '--num_exp',
                type=int, default=50,
                help='number of explanations to be evaluated for each test class (set None for every possible explanation)')
  parser.add_argument(
                '--verbose',
                type=int, default=0,
                help='set the level of additional details during the evaluation phase')
  parser.add_argument(
                '--n_rand_curves',
                type=int, default=5,
                help='number of random-generated insertion/deletion curves computed for each explanation (set 0 for no random curves)')
  args = parser.parse_args()
    
  CONFIG = args.config
  ROOT = args.root
  S_ROOT = args.s_root
  MAIN_DIR = args.main_dir
  CKP_PATH = args.ckp_path
  TEST_IMGS_PATH = args.test_imgs_path
  MODEL = args.model
  METHOD = args.method
  NUM_EXP = args.num_exp
  VERBOSE = args.verbose
  N_RAND_CURVES = args.n_rand_curves
  
  set_wdir(S_ROOT, CKP_PATH, MAIN_DIR, ROOT)
  
  from trainlib.trainlib import get_loaders
  from netlib.netlib import get_vgg_model, get_resnet_model, get_densenet_model, vgg16_ft
  from explib.explib import show_explanation, get_gradcam_results, create_global_heatmap
  from explib.rise import RISE
  from metricslib.metricslib import auc, gkern, CausalMetric
  from utilslib.utilslib import *
  
  if CONFIG == 'Scraping':
    config_file = 'class_config.json'
  else:
    config_file = 'class_config_VOC.json'
  
  with open(config_file) as json_file:  
    HOL_MER_DICT = json.load(json_file)
    
  run_evaluation(CONFIG, HOL_MER_DICT, S_ROOT, CKP_PATH, TEST_IMGS_PATH, MODEL, METHOD, verbose=VERBOSE, num_exp=NUM_EXP, n_rand_curves=N_RAND_CURVES)