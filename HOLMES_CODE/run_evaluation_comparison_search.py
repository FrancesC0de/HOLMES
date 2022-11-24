# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import warnings
from torch import nn
import cv2

"""**Set working directory**"""

def set_wdir(CKP_PATH, MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))

"""**Generate the explanations**"""

def generate_heatmaps(model_name, num_classes, test_images_path, meronyms, idx_to_class, pmode, p, checkpoint, printout=False, num_exp=None, size=(224, 224)):
  # for each image (key), save the img tensor, the holmes heatmap, the gradcam heatmap, and the predicted class (values)
  heatmaps = {}
  # get the desired filepaths
  test_imgs_paths = os.listdir(test_images_path)
  # filter out folders
  test_imgs_paths = [x for x in test_imgs_paths if os.path.isfile(os.path.join(test_images_path, x))]
  sel_test_imgs_paths = test_imgs_paths if num_exp is None else test_imgs_paths[:np.min([num_exp, len(test_imgs_paths)])]

  for test_img_path in tqdm(sel_test_imgs_paths):
    img_basepath = os.path.basename(test_img_path)
    complete_img_path = os.path.join(test_images_path, test_img_path)
    heatmaps[img_basepath] = {}
    # get the models to be used to generate grad-cam and holmes explanations
    model = get_model_by_name(model_name, hooks=False)
    model_hooks = get_model_by_name(model_name, hooks=True) # model with hooks
      
    # generate the HOLMES global heatmap
    drop_pred_values, meronym_explanations_dict = show_explanation(complete_img_path, model, meronyms, num_classes, idx_to_class, checkpoint, printout=printout, pmode=pmode, p=p)
    drop_values = [v[0] for v in drop_pred_values]
    pred_class_name = drop_pred_values[0][1]
    global_heatmap, _, _, _, _, _ = create_global_heatmap(meronyms, meronym_explanations_dict, drop_values)
    
    # generate the gradcam heatmap
    img_tensor, raw_heatmap, _, _ = get_gradcam_results(model_hooks, complete_img_path, class_idx=None, baseline=True, method='gradcam')

    # save into the dictionary
    heatmaps[img_basepath]['pred_class'] = pred_class_name
    heatmaps[img_basepath]['img_tensor'] = img_tensor
    heatmaps[img_basepath]['holmes_heatmap'] = global_heatmap
    heatmaps[img_basepath]['gradcam_heatmap'] = cv2.resize(raw_heatmap, size)

  return heatmaps
  
"""**Generate the insertion/deletion/preservation curves**"""

def compute_causal_metrics(model, img_tensor, final_heatmap, klen=11, ksig=5, size=224, verbose = 0):
    # Function that blurs input image
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2) 
    # Causal metrics
    insertion = CausalMetric(model, 'ins', size, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', size, substrate_fn=torch.zeros_like) 
    preservation = CausalMetric(model, 'pres', size, substrate_fn=torch.zeros_like) 
    # deletion
    h = deletion.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose)
    auc_deletion = auc(h)
    # insertion
    h = insertion.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose)
    auc_insertion = auc(h)
    # preservation
    h = preservation.single_run(img_tensor.cpu(), final_heatmap, verbose=verbose)
    auc_preservation = auc(h)

    return auc_deletion, auc_insertion, auc_preservation

def generate_curves(model_name, MERONYMS, test_images_path, heatmaps, verbose):
    # for each image (key), save the predicted class, the insertion, deletion, preservation curves for the holmes heatmap and for the gradcam heatmap
    curves = {}
    img_basepaths = heatmaps.keys()
    # get the model to be used to compute the causal metrics
    model = get_model_by_name(model_name, hooks=False)
    
    # generate the curves for each holonym image within the validation set
    for img_basepath in tqdm(img_basepaths):
      curves[img_basepath] = {}
      # get from the heatmaps dictionary the img tensor, the predicted class, the holmes and the gradcam heatmaps
      img_info = heatmaps[img_basepath]
      img_tensor = img_info['img_tensor']
      pred_class = img_info['pred_class']
      holmes_heatmap = img_info['holmes_heatmap']
      gradcam_heatmap = img_info['gradcam_heatmap']
      
      # compute causal metrics for the holmes global heatmap
      holmes_auc_deletion, holmes_auc_insertion, holmes_auc_preservation = compute_causal_metrics(model, img_tensor, holmes_heatmap)
      # compute causal metrics for the gradcam heatmap
      gradcam_auc_deletion, gradcam_auc_insertion, gradcam_auc_preservation = compute_causal_metrics(model, img_tensor, gradcam_heatmap)
      
      # save into the dictionary
      curves[img_basepath]['pred_class'] = pred_class
      curves[img_basepath]['holmes_deletion'] = holmes_auc_deletion
      curves[img_basepath]['holmes_insertion'] = holmes_auc_insertion
      curves[img_basepath]['holmes_preservation'] = holmes_auc_preservation
      curves[img_basepath]['gradcam_deletion'] = gradcam_auc_deletion
      curves[img_basepath]['gradcam_insertion'] = gradcam_auc_insertion
      curves[img_basepath]['gradcam_preservation'] = gradcam_auc_preservation
          
    return curves

  
"""**Evaluation**"""

def run_evaluation_comparison(CONFIG, HOL_MER_DICT, CKP_PATH, TEST_IMGS_PATH, MODEL, PMODE, P, verbose=0, num_exp=50):
  # first check if the path to the holonyms folder (e.g., the ImageNet validation sets folder) is correct
  if os.path.exists(TEST_IMGS_PATH) == False:
    raise Exception('Holonyms folder {} not found. Cannot proceed without it.'.format(TEST_IMGS_PATH))
  # dictionary to serialize
  curves_dict = {}
  # load data if already existing
  res_dir = "Results/grid_search"
  os.makedirs(res_dir, exist_ok=True)
  curves_json = os.path.join(res_dir, CONFIG + '_' + MODEL + '_' + PMODE + '_' + str(P) + '_curves.json')
  if os.path.exists(curves_json) == True:
    with open(curves_json) as json_file: 
      curves_dict = json.load(json_file)

  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    CHECKPOINT = os.path.join(CKP_PATH, (HOLONYM + '.pth'))
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    NUM_CLASSES = len(MERONYMS)
    IDX_TO_CLASS = { i : MERONYMS[i] for i in range(len(MERONYMS)) }
    # perform the evaluation on the holonym class only if not already done/serialized
    if HOLONYM not in curves_dict:
        # check if the path to the specific holonym test images exists
        try:
          test_images_path = os.path.join(TEST_IMGS_PATH, [x for x in os.listdir(TEST_IMGS_PATH) if (x.split(',')[0] == HOLONYM)][0])
          if os.path.exists(test_images_path) == False:
            raise Exception()
        except:
          raise Exception(('The ' + HOLONYM + ' holonym samples have not been found. Cannot generate their explanations without them.'))
          
        # generate both HOLMES and Grad-CAM heatmaps       
        heatmaps = generate_heatmaps(MODEL, NUM_CLASSES, test_images_path, MERONYMS, IDX_TO_CLASS, PMODE, P, checkpoint=CHECKPOINT, num_exp=num_exp)
        # generate the curves for each holonym image for both HOLMES and Grad-CAM heatmaps
        curves_stats = generate_curves(MODEL, MERONYMS, test_images_path, heatmaps, verbose)
        
        # update the dictionary
        curves_dict[HOLONYM] = {}
        curves_dict[HOLONYM]['parts'] = MERONYMS
        curves_dict[HOLONYM]['curves'] = curves_stats
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
                '--pmode',
                default='percentile',
                choices=['percentile', 'additive'],
                help='strategy employed to retain information over each meronym heatmap') 
  parser.add_argument(
                '--p_range',
                type=str, default="75-91",
                help='two numbers separated by a hyphen, representing the grid search range')      
  parser.add_argument(
                '--p_step',
                type=int, default=1,
                help='interval step applied upon the grid search range')
  parser.add_argument(
                '--num_exp',
                type=int, default=50,
                help='number of explanations to be evaluated for each test class (set None for every possible explanation)')
  parser.add_argument(
                '--verbose',
                type=int, default=0,
                help='set the level of additional details during the evaluation phase')
  args = parser.parse_args()
    
  CONFIG = args.config
  ROOT = args.root
  MAIN_DIR = args.main_dir
  CKP_PATH = args.ckp_path
  TEST_IMGS_PATH = args.test_imgs_path
  MODEL = args.model
  PMODE = args.pmode
  P_STEP = args.p_step
  P_RANGE = range(*list(map(int, args.p_range.split("-"))), P_STEP)
  NUM_EXP = args.num_exp
  VERBOSE = args.verbose
  
  set_wdir(CKP_PATH, MAIN_DIR, ROOT)
  
  from netlib.netlib import get_model_by_name
  from explib.explib import show_explanation, get_gradcam_results, create_global_heatmap
  from metricslib.metricslib import auc, gkern, CausalMetric
  from utilslib.utilslib import *
  
  if CONFIG == 'Scraping':
    config_file = 'class_config.json'
  else:
    config_file = 'class_config_VOC.json'
  
  with open(config_file) as json_file:  
    HOL_MER_DICT = json.load(json_file)
    
  for P in P_RANGE:
    run_evaluation_comparison(CONFIG, HOL_MER_DICT, CKP_PATH, TEST_IMGS_PATH, MODEL, PMODE, P, verbose=VERBOSE, num_exp=NUM_EXP)
