# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
import torch
from tqdm import tqdm

"""**Set working directory**"""

def set_wdir(S_ROOT, CKP_PATH, MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))
  # create non-existing directories
  if os.path.exists(S_ROOT) == False:
    # scraping dir
    os.mkdir(S_ROOT)
  if os.path.exists(CKP_PATH) == False:
    # checkpoints dir
    os.mkdir(CKP_PATH)

"""**Scraping**"""

def scrape(holonym, meronyms, root, meronyms_path, limit_g=40, limit_b=60, limit_vs=5, hamming_th=10, contamination=0.15):
  # quantities to serialize
  init_downs = []   # number of initial downloads
  num_dups = []     # duplicates number
  num_outliers = [] # outliers number
  final_downs = []  # number of deduplicates samples without outliers
  start = time.time()
  # save and retain the scraping arguments for later doing the visually similar images search
  arguments_list = scrape_parts(holonym, meronyms, root, limit_g=limit_g, limit_b=limit_b) 
  # for each meronym, image samples are scraped
  for i, m in enumerate(meronyms):
    image_dir = os.path.join(meronyms_path, m)
    # if the meronym folder doesn't exist, it is created
    if os.path.exists(image_dir) == False:
      os.mkdir(image_dir)
    # download the top visually similar images for each image
    similar_images_scraping(arguments_list[i], limit=limit_vs)
    # number of images initially downloaded
    init_downs_tmp = len(os.listdir(image_dir))
    init_downs.append(init_downs_tmp)
    # duplicates removal
    print('\nDuplicates removal for the meronym:', m)
    duplicates = delete_duplicates(image_dir, hamming_th)
    num_dups_tmp = len(duplicates)
    num_dups.append(num_dups_tmp)
    print('\nRemoved {} duplicates for the meronym: {}'.format(len(duplicates), m))
    # create the filtered list containing only the non-outlier filenames
    outputfile = os.path.join(meronyms_path, 'filtered_' + m + '.txt')
    # outliers removal
    print('\nOutlier removal for the meronym:', m)
    inliers = remove_outliers(image_dir, contamination=contamination)
    num_outliers_tmp = init_downs_tmp - num_dups_tmp - len(inliers)
    num_outliers.append(num_outliers_tmp)
    # create a file listing all the deduplicated inliers for the current meronym
    # in this way the outliers can be later filtered out
    errors = 0
    with open(outputfile, 'w') as list_file:
      for inlier in inliers:
        try:
            list_file.write(inlier + '\n')
        except:
            print("Skipping file : " + inlier)
            errors += 1
    # save the number of resulting deduplicated inliers
    final_downs_tmp = len(inliers) - errors
    final_downs.append(final_downs_tmp)
  end = time.time()
  scraping_time = (end - start)

  return scraping_time, init_downs, num_dups, num_outliers, final_downs
  
"""**Data loading**"""

def set_loaders(meronyms_path, batch_size, filterprefix = 'filtered_'):
  # create the train/validation/test dataloaders, w/o the outliers
  class_to_idx, trainloader, validloader, testloader = get_loaders(meronyms_path, batch_size, filterprefix)
  num_classes = len(class_to_idx)
  idx_to_class = {v: k for k, v in class_to_idx.items()}
  
  return trainloader, validloader, testloader, num_classes, idx_to_class
  
"""**Training**"""

def train_parts(num_classes, idx_to_class, checkpoint, trainloader, validloader, testloader, n_epochs=100, lr=1e-3, patience=5):
  model = vgg16_ft(num_classes).cuda()
  start = time.time()
  model = train(model, trainloader, validloader, checkpoint, num_classes=num_classes, n_epochs=n_epochs, lr=lr, max_epochs_stop=patience)
  end = time.time()
  training_time = (end - start)
  # test the model on the test set and save the final per class calibrated f1-score in the checkpoint
  f1s, _ = evaluate_all(testloader, model, num_classes, idx_to_class, training_time=training_time, save_file_name=checkpoint)

  return training_time, f1s
  
"""**Generate the explanations**"""
  
def generate_explanations(num_classes, test_images_path, meronyms, idx_to_class, checkpoint, printout=False, num_exp=None, p=90, score_t=0.10, metric_t=0.7):
  path_drop_dict = {}
  test_imgs_paths = os.listdir(test_images_path)
  if num_exp is None:
    sel_test_imgs_paths = test_imgs_paths
  else:
    sel_test_imgs_paths = test_imgs_paths[:num_exp]
  for test_img_path in tqdm(sel_test_imgs_paths):
    # get the original vgg16 model pretrained on ImageNet
    vgg_model = get_vgg_model()
    # generate the explanations
    drop_pred_values, meronym_explanations_dict = show_explanation(os.path.join(test_images_path, test_img_path), vgg_model, meronyms, num_classes, idx_to_class, checkpoint, printout=printout, score_t=score_t, metric_t=metric_t, p=p)
    base_path = test_img_path
    score_drop, pred_class = [values[0] for values in drop_pred_values], drop_pred_values[0][1]
    path_drop_dict[base_path] = {}
    path_drop_dict[base_path]['score_drop'] = score_drop
    path_drop_dict[base_path]['pred_class'] = pred_class

  return path_drop_dict, meronym_explanations_dict

"""**Main Pipeline**"""

def run_pipeline(HOL_MER_DICT, S_ROOT, CKP_PATH, TEST_IMGS_PATH, LIMIT_G, LIMIT_B, LIMIT_VS, HAMMING_TH, CONTAMINATION, N_EPOCHS, BATCH_SIZE, LR, PATIENCE, NUM_EXP, PRINTOUT, PERC, SCORE_T, METRIC_T):
  # first check if the path to the holonyms folder (e.g., the ImageNet validation sets folder) is correct
  if os.path.exists(TEST_IMGS_PATH) == False:
    raise Exception('Holonyms folder not found. Cannot proceed without it.')
  # dictionaries to serialize
  scrape_dict = {}
  train_dict = {}
  exp_dict = {}
  # load data if already existing
  if os.path.exists('scrape.json') == True:
    with open('scrape.json') as json_file: 
      scrape_dict = json.load(json_file)
  if os.path.exists('train.json') == True:
    with open('train.json') as json_file: 
      train_dict = json.load(json_file)
  if os.path.exists('exp.json') == True:
    with open('exp.json') as json_file: 
      exp_dict = json.load(json_file) 

  # execute the HOLMES pipeline for each selected holonym
  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    MERONYMS_PATH = os.path.join(S_ROOT, HOLONYM)
    NUM_CLASSES = len(MERONYMS)
    IDX_TO_CLASS = { i : MERONYMS[i] for i in range(len(MERONYMS)) }
    # for each meronym, create the associated folder where the samples will be downloaded
    for MERONYM in MERONYMS:
      m_path = os.path.join(MERONYMS_PATH, MERONYM)
      if os.path.exists(m_path) == False:
        os.makedirs(m_path, exist_ok=True)
    # scraping
    if HOLONYM not in scrape_dict:
      scraping_time, init_downs, num_dups, num_outliers, final_downs = scrape(HOLONYM, MERONYMS, S_ROOT, MERONYMS_PATH, LIMIT_G, LIMIT_B, LIMIT_VS, HAMMING_TH, CONTAMINATION)
      scrape_dict[HOLONYM] = {'parts': MERONYMS, 'scraping_time': scraping_time, 'init_downs': init_downs, 'num_outliers': num_outliers, 'num_dups': num_dups, 'final_downs' : final_downs}
      # serialization
      with open('scrape.json', 'w') as fp:
        json.dump(scrape_dict, fp)
    # training 
    CHECKPOINT = os.path.join(CKP_PATH, (HOLONYM + '.pth'))
    if HOLONYM not in train_dict:
      # data loading
      trainloader, validloader, testloader, _, _ = set_loaders(MERONYMS_PATH, BATCH_SIZE)
      # do training only if not already previously done (no checkpoint or no f1-scores calculated)
      if os.path.exists(CHECKPOINT) == False or 'f1score_c' not in torch.load(CHECKPOINT):
        # training
        training_time, f1s = train_parts(NUM_CLASSES, IDX_TO_CLASS, CHECKPOINT, trainloader, validloader, testloader, 
                                            n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE)
      # checkpoint and scores exist but have not been serialized                                     
      else:
        ckp = torch.load(CHECKPOINT)
        training_time = ckp['training_time']
        f1s = ckp['f1score_c'] 
      train_dict[HOLONYM] = {'parts': MERONYMS, 'training_time': training_time, 'f1_scores': f1s}
      # serialization
      with open('train.json', 'w') as fp:
        json.dump(train_dict, fp)
    # explanations
    if HOLONYM not in exp_dict:
      # check if the path to the specific holonym test images exists
      try:
        test_images_path = os.path.join(TEST_IMGS_PATH, [x for x in os.listdir(TEST_IMGS_PATH) if x.startswith(HOLONYM)][0])
        if os.path.exists(test_images_path) == False:
          raise Exception()
      except:
        raise Exception(('The ' + HOLONYM + ' holonym samples have not been found. Cannot generate their explanations without them.'))
      path_drop_dict, _ = generate_explanations(NUM_CLASSES, test_images_path, MERONYMS, IDX_TO_CLASS, checkpoint=CHECKPOINT, printout=PRINTOUT, num_exp=NUM_EXP, p=PERC, score_t=SCORE_T, metric_t=METRIC_T)
      path_drop_dict['parts'] = MERONYMS
      exp_dict[HOLONYM] = path_drop_dict
      # serialization
      with open('exp.json', 'w') as fp:
        json.dump(exp_dict, fp)

"""**Main**"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
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
                '--limit_g',
                type=int, default=40,
                help='number of images to scrape with the google engine')
  parser.add_argument(
                '--limit_b',
                type=int, default=60,
                help='number of images to scrape with the bing engine')
  parser.add_argument(
                '--limit_vs',
                type=int, default=5,
                help='number of visually similar image to download for each previously scraped image')
  parser.add_argument(
                '--hamming_th',
                type=float, default=10,
                help='hamming distance threshold for the pHash algorithm')
  parser.add_argument(
                '--contamination',
                type=float, default=0.15,
                help='expected percentage of outliers')
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

  # directories
  ROOT = args.root
  S_ROOT = args.s_root
  MAIN_DIR = args.main_dir
  CKP_PATH = args.ckp_path
  TEST_IMGS_PATH = args.test_imgs_path
  # scraping
  LIMIT_G = args.limit_g
  LIMIT_B = args.limit_b
  LIMIT_VS = args.limit_vs
  HAMMING_TH = args.hamming_th
  CONTAMINATION = args.contamination
  # training
  N_EPOCHS = args.n_epochs
  BATCH_SIZE = args.batch_size
  LR = args.lr
  PATIENCE = args.patience
  # explanations
  NUM_EXP = args.num_exp
  PRINTOUT = args.printout
  PERC = args.perc
  SCORE_T = args.score_t
  METRIC_T = args.metric_t

  set_wdir(S_ROOT, CKP_PATH, MAIN_DIR, ROOT)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

  from scrapelib.scrapelib import scrape_parts, similar_images_scraping
  from duplib.duplib import delete_duplicates
  from outlib.outlib import remove_outliers
  from trainlib.trainlib import get_loaders, train, evaluate_all
  from netlib.netlib import vgg16_ft, get_vgg_model
  from explib.explib import show_explanation

  with open('class_config.json') as json_file: 
    HOL_MER_DICT = json.load(json_file)
  run_pipeline(HOL_MER_DICT, S_ROOT, CKP_PATH, TEST_IMGS_PATH, LIMIT_G, LIMIT_B, LIMIT_VS, HAMMING_TH, CONTAMINATION, N_EPOCHS, BATCH_SIZE, LR, PATIENCE, NUM_EXP, PRINTOUT, PERC, SCORE_T, METRIC_T) 
