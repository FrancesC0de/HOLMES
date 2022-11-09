# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
import torch
from tqdm import tqdm

"""**Set working directory**"""

def set_wdir(CKP_PATH, MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))
  # create non-existing directories
  if os.path.exists(CKP_PATH) == False:
    # checkpoints dir
    os.mkdir(CKP_PATH)
  # logging dirs
  loss_dir = os.path.join(CKP_PATH, 'Loss')
  if os.path.exists(loss_dir) == False:
      os.mkdir(loss_dir)
  perf_dir = os.path.join(CKP_PATH, 'Performance')
  if os.path.exists(perf_dir) == False:
      os.mkdir(perf_dir)
    
"""**Data loading**"""

def set_loaders(meronyms_path, batch_size, filterprefix = 'filtered_'):
  # create the train/validation/test dataloaders, w/o the outliers
  class_to_idx, trainloader, validloader, testloader = get_loaders(meronyms_path, batch_size, filterprefix)
  num_classes = len(class_to_idx)
  idx_to_class = {v: k for k, v in class_to_idx.items()}
  
  return trainloader, validloader, testloader, num_classes, idx_to_class
  
"""**Training**"""

def train_parts(model_name, num_classes, idx_to_class, checkpoint, trainloader, validloader, testloader, n_epochs=100, lr=1e-3, patience=5):
  if model_name == 'vgg16':
    model = vgg16_ft(num_classes).cuda()
  elif model_name == 'resnet50':
    model = resnet50_ft(num_classes).cuda()
  elif model_name == 'densenet121':
    model = densenet121_ft(num_classes).cuda()
  elif model_name == 'mobilenet':
    model = mobilenet_ft(num_classes).cuda()
  elif model_name == 'inception':
    model = inception_ft(num_classes).cuda()
  elif model_name == 'deit':
    model = deit_ft(num_classes).cuda()
  else:
    raise Exception('Model {} not supported.'.format(model_name))
  start = time.time()
  model = train(model, trainloader, validloader, checkpoint, num_classes=num_classes, n_epochs=n_epochs, lr=lr, max_epochs_stop=patience)
  end = time.time()
  training_time = (end - start)
  # test the model on the test set and save the final per class calibrated f1-score in the checkpoint
  f1s, test_acc = evaluate_all(testloader, model, num_classes, idx_to_class, training_time=training_time, save_file_name=checkpoint)

  return training_time, f1s, test_acc
  
"""**Run Training Step**"""

def run_training_only(CONFIG, HOL_MER_DICT, S_ROOT, FILTER_PREFIX, CKP_PATH, MODEL, N_EPOCHS, BATCH_SIZE, LR, PATIENCE):
  # dictionary to serialize
  train_dict = {}
  # load data if already existing
  train_json = CONFIG + '_' + MODEL + '_train.json'
  if os.path.exists(train_json) == True:
    with open(train_json) as json_file: 
      train_dict = json.load(json_file)

  # execute the HOLMES training step for each selected holonym
  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    MERONYMS_PATH = os.path.join(S_ROOT, HOLONYM)
    # training 
    CHECKPOINT = os.path.join(CKP_PATH, (HOLONYM + '.pth'))
    if HOLONYM not in train_dict:
      # data loading
      trainloader, validloader, testloader, NUM_CLASSES, IDX_TO_CLASS = set_loaders(MERONYMS_PATH, BATCH_SIZE, filterprefix = (None if FILTER_PREFIX else 'filtered_'))
      # do training only if not already previously done (no checkpoint or no f1-scores calculated)
      if os.path.exists(CHECKPOINT) == False or 'f1score_c' not in torch.load(CHECKPOINT):
        # training
        training_time, f1s, test_acc = train_parts(MODEL, NUM_CLASSES, IDX_TO_CLASS, CHECKPOINT, trainloader, validloader, testloader, 
                                            n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE)
      # checkpoint and scores exist but have not been serialized                                     
      else:
        ckp = torch.load(CHECKPOINT)
        training_time = ckp['training_time']
        f1s = ckp['f1score_c'] 
        test_acc = ckp['test_accuracy']
      train_dict[HOLONYM] = {'parts': MERONYMS, 'training_time': training_time, 'f1_scores': f1s, 'test_acc': test_acc}
      # serialization
      with open(train_json, 'w') as fp:
        json.dump(train_dict, fp, indent=4)
        
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
                '--filter_prefix',
                action='store_true',
                help='use <filtered_> prefix to filter out unwanted scraped images')
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
                default='HOLMES_CODE',
                help='partial path from the root (e.g /My Drive/) to the HOLMES folder')
  parser.add_argument(
                '--model',
                default='vgg16',
                help='CNN model to be explained')
  parser.add_argument(
                '--n_epochs',
                type=int, default=100,
                help='number of epochs to train each auxiliary model')
  parser.add_argument(
                '--batch_size',
                type=int, default=64,
                help='batch size training hyper-parameter')
  parser.add_argument(
                '--lr',
                type=float, default=0.001,
                help='learning rate training hyper-parameter')
  parser.add_argument(
                '--patience',
                type=int, default=5,
                help='Early Stopping training hyper-parameter')
  args = parser.parse_args()
  
  # configuration
  CONFIG = args.config
  # directories
  ROOT = args.root
  S_ROOT = args.s_root
  FILTER_PREFIX = args.filter_prefix
  MAIN_DIR = args.main_dir
  CKP_PATH = args.ckp_path
  # training
  MODEL = args.model
  N_EPOCHS = args.n_epochs
  BATCH_SIZE = args.batch_size
  LR = args.lr
  PATIENCE = args.patience
  
  set_wdir(CKP_PATH, MAIN_DIR, ROOT)
  
  from trainlib.trainlib import get_loaders, train, evaluate_all
  from netlib.netlib import vgg16_ft, resnet50_ft, densenet121_ft, mobilenet_ft, inception_ft
  from netlib.transformers import deit_ft
  
  if CONFIG == 'Scraping':
    config_file = 'class_config.json'
  else:
    config_file = 'class_config_VOC.json'
  with open(config_file) as json_file: 
    HOL_MER_DICT = json.load(json_file)
  run_training_only(CONFIG, HOL_MER_DICT, S_ROOT, FILTER_PREFIX, CKP_PATH, MODEL, N_EPOCHS, BATCH_SIZE, LR, PATIENCE) 