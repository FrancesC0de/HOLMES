"""Prepare the ImageNet dataset"""
import os
import sys
import pickle
import gzip
import tarfile
from tqdm import tqdm

def extract_val(tar_fname):
    print('Extracting ' + tar_fname + '...')
    with tarfile.open(tar_fname) as tar:
        # Go over each member
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member)
    # move images to proper subfolders
    val_maps_file = 'imagenet_val_maps.pklz'
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(d)
    for m in mappings:
        os.rename(m[0], os.path.join(m[1], m[0]))
        
def get_arg(pos, default=None):
    try:
        return sys.argv[pos]
    except IndexError:
        return default

if __name__ == '__main__':
    val_tar_fname = get_arg(1, 'ILSVRC2012_img_val.tar')
    extract_val(val_tar_fname)
    print('Renaming folders...')
    with open('../synset_words.txt') as f:
        for line in f:
            words = line.split(" ", 1)
            id = words[0]
            label = words[1].rstrip('\n')
            if os.path.exists(id):
              #print(id + ' exists')
              if not os.path.exists(label):
                #print(label + ' label does not exist, renaming subfolder...')
                os.rename(id,label)
            else:
              print(id + ' not existing (either absent or already renamed')