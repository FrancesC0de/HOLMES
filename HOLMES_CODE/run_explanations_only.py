# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm

"""**Set working directory**"""

def set_wdir(MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))
    
"""**Generate the explanations**"""
  
def generate_explanations(model_name, num_classes, test_images_path, meronyms, idx_to_class, checkpoint, printout=False, num_exp=None, score_t=0.10, metric_t=0.7, p=90):
  path_drop_dict = {}
  test_imgs_paths = os.listdir(test_images_path)
  # filter out folders
  test_imgs_paths = [x for x in test_imgs_paths if os.path.isfile(os.path.join(test_images_path, x))]
  if num_exp is None:
    sel_test_imgs_paths = test_imgs_paths
  else:
    sel_test_imgs_paths = test_imgs_paths[:num_exp]
  for test_img_path in tqdm(sel_test_imgs_paths):
    # get the original model pretrained on ImageNet
    if model_name == 'vgg16':
      model = get_vgg_model()
    elif model_name == 'resnet50':
      model = get_resnet_model()
    elif model_name == 'densenet121':
      model = get_densenet_model()
    else:
      raise Exception('Model {} not supported.'.format(model_name))
    # generate the explanations
    drop_pred_values, meronym_explanations_dict = show_explanation(os.path.join(test_images_path, test_img_path), model, meronyms, num_classes, idx_to_class, checkpoint, printout=printout, score_t=score_t, metric_t=metric_t, p=p)
    base_path = test_img_path
    score_drop, pred_class = [values[0] for values in drop_pred_values], drop_pred_values[0][1]
    path_drop_dict[base_path] = {}
    path_drop_dict[base_path]['score_drop'] = score_drop
    path_drop_dict[base_path]['pred_class'] = pred_class

  return path_drop_dict, meronym_explanations_dict
  
"""**Run Explanations Step**"""

def run_explanations_only(CONFIG, HOL_MER_DICT, CKP_PATH, TEST_IMGS_PATH, MODEL, NUM_EXP, PRINTOUT, PERC, SCORE_T, METRIC_T):
  # first check if the path to the holonyms folder (e.g., the ImageNet validation sets folder) is correct
  if os.path.exists(TEST_IMGS_PATH) == False:
    raise Exception('Holonyms folder not found. Cannot proceed without it.')
  # dictionary to serialize
  exp_dict = {}
  # load data if already existing
  exp_json = CONFIG + '_' + MODEL + '_exp.json'
  if os.path.exists(exp_json) == True:
    with open(exp_json) as json_file: 
      exp_dict = json.load(json_file) 

  # execute the HOLMES explanations step for each selected holonym
  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    NUM_CLASSES = len(MERONYMS)
    IDX_TO_CLASS = { i : MERONYMS[i] for i in range(len(MERONYMS)) }
    # explanations
    if HOLONYM not in exp_dict:
      # check if the path to the specific holonym test images exists
      try:
        test_images_path = os.path.join(TEST_IMGS_PATH, [x for x in os.listdir(TEST_IMGS_PATH) if (x.split(',')[0] == HOLONYM)][0])
        if os.path.exists(test_images_path) == False:
          raise Exception()
      except:
        raise Exception(('The ' + HOLONYM + ' holonym samples have not been found. Cannot generate their explanations without them.'))
      CHECKPOINT = os.path.join(CKP_PATH, (HOLONYM + '.pth'))
      path_drop_dict, _ = generate_explanations(MODEL, NUM_CLASSES, test_images_path, MERONYMS, IDX_TO_CLASS, checkpoint=CHECKPOINT, printout=PRINTOUT, num_exp=NUM_EXP, score_t=SCORE_T, metric_t=METRIC_T, p=PERC)
      path_drop_dict['parts'] = MERONYMS
      exp_dict[HOLONYM] = path_drop_dict
      # serialization
      with open(exp_json, 'w') as fp:
        json.dump(exp_dict, fp)
       
"""**Main**"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
                '--config',
                default='Scraping',
                choices=['Scraping', 'VOC'],
                help='load the configuration file for the scraping or the VOC setting')
  parser.add_argument(
                '--test_imgs_path',
                default='Holonyms',
                help='folder where the test images for the holonyms are stored')
  parser.add_argument(
                '--ckp_path',
                default='Checkpoints',
                help='folder where model checkpoints are saved')
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
                type=int, default=None,
                help='number of explanations to provide for each test class (set None for every possible explanation)')
  parser.add_argument(
                '--printout',
                action='store_true',
                help='decide whether showing the explanations or not')
  parser.add_argument(
                '--perc',
                type=int, default=90,
                help='percentile to compute over any meronym heatmap')  
  parser.add_argument(
                '--score_t',
                type=float, default=10.0,
                help='threshold to be applied on the holonym score drop percentage for retaining an explanation')
  parser.add_argument(
                '--metric_t',
                type=float, default=0.7,
                help='threshold to be applied on the chosen metric (calibrated F1-score by default) for retaining an explanation')
  args = parser.parse_args()
  
  # configuration
  CONFIG = args.config
  # directories
  ROOT = args.root
  MAIN_DIR = args.main_dir
  CKP_PATH = args.ckp_path
  TEST_IMGS_PATH = args.test_imgs_path
  # explanations
  MODEL = args.model
  NUM_EXP = args.num_exp
  PRINTOUT = args.printout
  PERC = args.perc
  SCORE_T = args.score_t
  METRIC_T = args.metric_t

  set_wdir(MAIN_DIR, ROOT)
  
  from netlib.netlib import get_vgg_model, get_resnet_model, get_densenet_model
  from explib.explib import show_explanation
  
  if CONFIG == 'Scraping':
    config_file = 'class_config.json'
  else:
    config_file = 'class_config_VOC.json'
  with open(config_file) as json_file: 
    HOL_MER_DICT = json.load(json_file)
  run_explanations_only(CONFIG, HOL_MER_DICT, CKP_PATH, TEST_IMGS_PATH, MODEL, NUM_EXP, PRINTOUT, PERC, SCORE_T, METRIC_T)