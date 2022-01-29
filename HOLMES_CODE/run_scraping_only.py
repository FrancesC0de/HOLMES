# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
from tqdm import tqdm

"""**Set working directory**"""

def set_wdir(S_ROOT, MAIN_DIR='HOLMES', ROOT='../'):
  # if I'm not already in the HOLMES directory, set it as working dir
  if (os.path.join(ROOT,MAIN_DIR)) not in os.getcwd():
    os.chdir(os.path.join(ROOT,MAIN_DIR))
  # create non-existing directories
  if os.path.exists(S_ROOT) == False:
    # scraping dir
    os.mkdir(S_ROOT)
    
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
  
"""**Run Image Scraping Step**"""

def run_scraping_only(HOL_MER_DICT, S_ROOT, LIMIT_G, LIMIT_B, LIMIT_VS, HAMMING_TH, CONTAMINATION):
  # dictionary to serialize
  scrape_dict = {}
  # load data if already existing
  if os.path.exists('scrape.json') == True:
    with open('scrape.json') as json_file: 
      scrape_dict = json.load(json_file)

  # execute the HOLMES image scraping step for each selected holonym
  for HOLONYM in HOL_MER_DICT.keys():
    print('\n' + HOLONYM.upper() + '\n')
    # set the meronyms variables and paths for the current holonym
    MERONYMS = sorted(HOL_MER_DICT[HOLONYM])
    MERONYMS_PATH = os.path.join(S_ROOT, HOLONYM)
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
                '--root',
                default='../',
                help='root of the HOLMES main directory')
  parser.add_argument(
                '--main_dir',
                default='HOLMES_DSAA2021',
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
  args = parser.parse_args()
  
  # directories
  ROOT = args.root
  S_ROOT = args.s_root
  MAIN_DIR = args.main_dir
  # scraping
  LIMIT_G = args.limit_g
  LIMIT_B = args.limit_b
  LIMIT_VS = args.limit_vs
  HAMMING_TH = args.hamming_th
  CONTAMINATION = args.contamination
  
  set_wdir(S_ROOT, MAIN_DIR, ROOT)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  
  from scrapelib.scrapelib import scrape_parts, similar_images_scraping
  from duplib.duplib import delete_duplicates
  from outlib.outlib import remove_outliers
  
  with open('class_config.json') as json_file: 
    HOL_MER_DICT = json.load(json_file)
  run_scraping_only(HOL_MER_DICT, S_ROOT, LIMIT_G, LIMIT_B, LIMIT_VS, HAMMING_TH, CONTAMINATION) 