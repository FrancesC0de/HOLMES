# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import warnings

"""**Set working directory**"""

def set_wdir(CKP_PATH, MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))
        
def get_binary_masks(MERONYMS, img_path, json_ann_folder):
    img_base_path = os.path.basename(img_path)
    json_base_path = img_base_path[:-4] + ".json"
    # assume the annotations folder is within the holonym folder
    json_ann_path = os.path.join(img_path[:-(len(img_base_path))], json_ann_folder, json_base_path)
    # load the bboxes gts from the json file
    with open(json_ann_path) as json_f:
        bboxes = json.load(json_f)
    # retrieve the image 
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape 
    # start generating masks
    masks = []
    for i, meronym in enumerate(MERONYMS):
        img_mask = np.ones_like(cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY))
        m_bboxes = bboxes[meronym]
        img = image.copy()  
        for m_bbox in m_bboxes:
            x_min, y_min, x_max, y_max = m_bbox
            # binarize the bbox annotation
            for i in range(y_min, y_max):
                img_mask[i,x_min:x_max] = 0    
        masks.append(img_mask)
    
    return masks # return meronym masks sized as the original image resolution
    
    
"""**Generate the statistics needed for the evaluation**"""

def generate_holonym_stats(model_name, num_classes, test_images_path, annotations_folder, meronyms, idx_to_class, checkpoint, printout=False, num_exp=None, class_names_path='./explib/class_names.json'):
  # get the desired filepaths
  test_imgs_paths = os.listdir(test_images_path)
  # filter out folders
  test_imgs_paths = [x for x in test_imgs_paths if os.path.isfile(os.path.join(test_images_path, x))]
  sel_test_imgs_paths = test_imgs_paths if num_exp is None else test_imgs_paths[:np.min([num_exp, len(test_imgs_paths)])]
  # load the idx to classname imagenet dictionary
  with open(class_names_path) as json_file: 
    class_names = json.load(json_file)        

  stats = {}
  for test_img_path in tqdm(sel_test_imgs_paths):
    img_basepath = os.path.basename(test_img_path)
    complete_img_path = os.path.join(test_images_path, test_img_path)
    stats[img_basepath] = {}
    # get the models to be used to generate grad-cam and holmes explanations
    model = get_model_by_name(model_name, hooks=False)
    model_hooks = get_model_by_name(model_name, hooks=True) # model with hooks

    # generate HOLMES meronyms explanations
    drop_pred_values, meronym_explanations_dict = show_explanation(complete_img_path, model, meronyms, num_classes, idx_to_class, checkpoint, printout=printout)
    pred_class_name = drop_pred_values[0][1]
    stats[img_basepath]['pred_class'] = pred_class_name
    # get the binary masks of each meronym bounding box for the image 
    bbox_masks = get_binary_masks(meronyms, complete_img_path, annotations_folder)
    # get the continuous meronyms heatmaps normalized between 0 and 1 
    normalized_heatmaps = [(meronym_explanations_dict[meronym]['norm_hm']) for meronym in meronyms]
    # calculate the per-pixel AUC score
    auc_scores_holmes = [roc_auc_score(np.logical_not(bbox_masks[i]).ravel(), normalized_heatmaps[i].ravel()) for i in range(num_classes)]
    stats[img_basepath]['HOLMES_AUCs'] = auc_scores_holmes
      
    # generate the Grad-CAM explanation for the whole (holonym) image 
    img_tensor, raw_heatmap, _, _ = get_gradcam_results(model_hooks, complete_img_path, class_idx = None, baseline = True, method = 'gradcam')

    # calculate AUC obtained with the whole Grad-Cam heatmap
    auc_scores_gcam = [roc_auc_score(np.logical_not(bbox_masks[i]).ravel(), raw_heatmap.ravel()) for i in range(num_classes)]
    stats[img_basepath]['GRADCAM_AUCs'] = auc_scores_gcam

  return stats
  

"""**Evaluation**"""

def run_evaluation(CONFIG, HOL_MER_DICT, CKP_PATH, TEST_IMGS_PATH, ANNOTATIONS_FOLDER, MODEL, num_exp=50):
  # first check if the path to the holonyms folder (e.g., the ImageNet validation sets folder) is correct
  if os.path.exists(TEST_IMGS_PATH) == False:
    raise Exception('Holonyms folder not found. Cannot proceed without it.')
  # dictionary to serialize
  eval_dict = {}
  # load data if already existing
  eval_json = CONFIG + '_' + MODEL + '_evaluation.json'
  if os.path.exists(eval_json) == True:
    with open(eval_json) as json_file: 
      eval_dict = json.load(json_file)

  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    CHECKPOINT = os.path.join(CKP_PATH, (HOLONYM + '.pth'))
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    NUM_CLASSES = len(MERONYMS)
    IDX_TO_CLASS = { i : MERONYMS[i] for i in range(len(MERONYMS)) }
    # perform the evaluation on the holonym class only if not already done/serialized
    if HOLONYM not in eval_dict:
        # check if the path to the specific holonym test images exists
        try:
          test_images_path = os.path.join(TEST_IMGS_PATH, [x for x in os.listdir(TEST_IMGS_PATH) if (x.split(',')[0] == HOLONYM)][0])
          if os.path.exists(test_images_path) == False:
            raise Exception()
        except:
          raise Exception(('The ' + HOLONYM + ' holonym samples have not been found. Cannot generate their explanations without them.'))
        # generate explanations according to the selected XAI method
        holonym_stats = generate_holonym_stats(MODEL, NUM_CLASSES, test_images_path, ANNOTATIONS_FOLDER, MERONYMS, IDX_TO_CLASS, checkpoint=CHECKPOINT, num_exp=num_exp)

        # update the dictionary
        eval_dict[HOLONYM] = {}
        eval_dict[HOLONYM]['parts'] = MERONYMS
        eval_dict[HOLONYM]['stats'] = holonym_stats
        # serialization
        with open(eval_json, 'w') as fp:
          json.dump(eval_dict, fp)

"""**Main**"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
                '--config',
                default='VOC',
                choices=['Scraping', 'VOC'],
                help='load the configuration file for the scraping or the VOC setting')
  parser.add_argument(
                '--ckp_path',
                default='Checkpoints',
                help='folder where model checkpoints are saved')
  parser.add_argument(
                '--test_imgs_path',
                default='VOC/TEST',
                help='folder where the test images for the holonyms are stored')
  parser.add_argument(
                '--annotations_folder',
                default='VOC_Test',
                help='folder where the bounding box annotation files are stored')
  parser.add_argument(
                '--root',
                default='../',
                help='root of the HOLMES main directory')
  parser.add_argument(
                '--main_dir',
                default='HOLMES_DSAA2021',
                help='partial path from the root (e.g /My Drive/) to the HOLMES folder')
  parser.add_argument(
                '--model',
                default='vgg16',
                help='CNN model to be explained')
  parser.add_argument(
                '--num_exp',
                type=int, default=50,
                help='number of explanations to be evaluated for each test class (set None for every possible explanation)')
  args = parser.parse_args()
    
  CONFIG = args.config
  ROOT = args.root
  MAIN_DIR = args.main_dir
  CKP_PATH = args.ckp_path
  TEST_IMGS_PATH = args.test_imgs_path
  ANNOTATIONS_FOLDER = args.annotations_folder
  MODEL = args.model
  NUM_EXP = args.num_exp
  
  set_wdir(CKP_PATH, MAIN_DIR, ROOT)
  
  from netlib.netlib import get_model_by_name
  from explib.explib import show_explanation, get_gradcam_results
  from utilslib.utilslib import *
  
  if CONFIG == 'Scraping':
    config_file = 'class_config.json'
  else:
    config_file = 'class_config_VOC.json'
  
  with open(config_file) as json_file:  
    HOL_MER_DICT = json.load(json_file)
    
  run_evaluation(CONFIG, HOL_MER_DICT, CKP_PATH, TEST_IMGS_PATH, ANNOTATIONS_FOLDER, MODEL, num_exp=NUM_EXP)