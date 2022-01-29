# -*- coding: utf-8 -*-

from utilslib.utilslib import *
from metricslib.metricslib import *
from netlib.netlib import load_best_model

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFile

import os
import torchvision
from torch import optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Subset

from copy import copy, deepcopy
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""**Default Arguments**"""

NUM_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 5

"""**Load the meronyms' dataset(s)**"""

imagenet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

# data augmentation
aug_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])])

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset(directory: str, filterprefix: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    skipped_files = 0
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        # open the filtered file, e.g. filtered_wheel.txt, and get the filenames
        with open(os.path.join(directory, (filterprefix + target_class + '.txt'))) as my_file:
          for line in my_file:
            item = line.rstrip('\n'), class_index
            # check if pil is able to open the image
            try:
              img = Image.open(item[0])
              instances.append(item)
            except Exception as e:
              skipped_files += 1
              
    if skipped_files != 0:
        print('# Files skipped: {}'.format(skipped_files)) 

    return instances

class MDataset(VisionDataset):
  def __init__(self, root, filterprefix, transform=None, target_transform=None):
      self.root = root
      self.filterprefix = filterprefix
      self.transform = transform
      self.target_transform = target_transform

      classes, class_to_idx = self._find_classes(self.root)
      samples = make_dataset(self.root, self.filterprefix, class_to_idx)

      self.loader = pil_loader
      self.classes = classes
      self.class_to_idx = class_to_idx
      self.samples = samples
      self.targets = [s[1] for s in samples]

  def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
      classes = [d.name for d in os.scandir(dir) if d.is_dir()]
      classes.sort()
      class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
      return classes, class_to_idx

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
      path, target = self.samples[index]
      sample = self.loader(path)
      if self.transform is not None:
          sample = self.transform(sample)
      if self.target_transform is not None:
          target = self.target_transform(target)

      return sample, target

  def __len__(self) -> int:
      return len(self.samples)

def train_val_test_loading(dataset, batch_size, test_split=0.1, val_split=0.1):
    y = dataset.targets
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, stratify=y, random_state=42)
    y = [dataset.targets[i] for i in train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=val_split, stratify=y, random_state=42)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['train'].dataset = copy(dataset)
    datasets['train'].dataset.transform = aug_transform

    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=(True if x == 'train' else False), num_workers=0)
              for x in ['train', 'val', 'test']}
    return dataloaders

def get_loaders(meronyms_path, batch_size=BATCH_SIZE, filterprefix=None):
  if filterprefix is None:
    dataset = torchvision.datasets.ImageFolder(meronyms_path, transform = imagenet_transform)
  else:
    dataset = MDataset(meronyms_path, filterprefix, transform = imagenet_transform)

  dataloaders = train_val_test_loading(dataset, batch_size)

  return dataset.class_to_idx, dataloaders['train'], dataloaders['val'], dataloaders['test']

"""**Training**"""

def show_train_results(history, path, name):
  # loss
  plt.figure(figsize=(8, 6))

  for c in ['train_loss', 'valid_loss']:
      plt.plot(
          history[c], label=c)
  plt.xticks(range(len(history[c])))
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Losses')
  # save inside <Chekpointfolder> / <Loss> / <Holonymname>
  plt.savefig(os.path.join(path, 'Loss', (name + '.png')))
  plt.close()

  # accuracy
  plt.figure(figsize=(8, 6))
  for c in ['train_acc', 'valid_acc']:
      plt.plot(
          100 * history[c], label=c)
  plt.xticks(range(len(history[c])))
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training and Validation Accuracy')
  # save inside <Chekpointfolder> / <Performance> / <Holonymname>
  plt.savefig(os.path.join(path, 'Performance', (name + '.png')))
  plt.close()

def train(model, train_loader, valid_loader, save_file_name, num_classes, n_epochs=NUM_EPOCHS, 
            lr=LR, max_epochs_stop=PATIENCE, print_every=1, save_hist=True):
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []
    for epoch in range(n_epochs):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()

        # Training loop
        for ii, (data, target) in enumerate(tqdm(train_loader)):
            # Tensors to gpu
            data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

        # After training loops ends, start validation
        else:
            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'Epoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_max_acc = valid_acc
                    best_epoch = epoch
                    
                    if save_file_name is not None:
                      # Save model classifier
                      torch.save({
                        'model_state_dict': model.classifier.state_dict(),
                        }, save_file_name)

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )

                        break

    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )

    if save_hist == True:
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc']) 
        # Serialize training loss and performance
        ckp_basepath = os.path.basename(save_file_name)
        ckpfolder_path = save_file_name[:-len(ckp_basepath)]
        show_train_results(history, ckpfolder_path, ckp_basepath.split('.')[0])
        
    model = load_best_model(model.name, num_classes, save_file_name=save_file_name)
    return model

"""**Inference**"""
  
def evaluate_all(test_loader, model, n_classes, IDX_TO_CLASS, training_time = None, save_file_name = None):
  test_acc = 0.0
  # lists to put together predictions and true labels
  target_list = []
  pred_list = []
  # Set to evaluation
  with torch.no_grad():
      model.eval()
      for data, target in test_loader:
          # Tensors to gpu
          data, target = data.cuda(), target.cuda()
          target_list.extend(target.tolist())
          # Model outputs 
          output = model(data)

          # Get predictions
          _, pred = torch.max(output, dim=1)
          pred_list.extend(pred.tolist())
          
          # compare predictions to true label
          correct_tensor = pred.eq(target.data.view_as(pred))

          # calculate accuracy for all classes together
          accuracy = torch.mean(
              correct_tensor.type(torch.FloatTensor))
          # Multiply average accuracy times the number of examples
          test_acc += accuracy.item() * data.size(0)
          
  # calibrated F1-score (per part)
  f1score_c = compute_calibrated_f1score(target_list, np.array(pred_list), n_classes)
  f1score_c = list(f1score_c.values())
  print('Calibrated F1-scores: ' +str([round(score, 3) for score in f1score_c]))
          
  # Calculate average accuracy
  test_acc = test_acc / len(test_loader.dataset)

  print(f'Total Test Accuracy: {100 * test_acc:.2f}')
  
  # save everything inside the checkpoint
  checkpoint = torch.load(save_file_name)
  torch.save({
            'model_state_dict': checkpoint['model_state_dict'],
            'test_accuracy': test_acc,
            'f1score_c': f1score_c,
            'training_time': training_time,
            }, save_file_name)

  return f1score_c