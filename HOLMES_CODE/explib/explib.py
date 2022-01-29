# -*- coding: utf-8 -*-

from utilslib.utilslib import *
from netlib.netlib import *

import cv2
import json
import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image, ImageFile
from copy import copy
from sklearn.preprocessing import normalize

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from cv2 import imshow as cv2_imshow
from copy import copy, deepcopy

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""**Show the image to be explained**"""

def show_test_image(test_image_path):
  img = cv2.imread(test_image_path)
  img = cv2.resize(img, (224, 224))
  cv2_imshow('', img) 

"""**XAI functions**"""

mean = [0.485, 0.456, 0.406] 
R = int(round((mean[0] * 255)))
G = int(round((mean[1] * 255)))
B = int(round((mean[2] * 255)))

def get_activation_weights(method, activations, gradients, score):
  if method == 'gradcam':
    # pool the gradients across the channels (feature maps) 
    # gradients flowing back are global-average-pooled over the width and height dimensions
    weights = torch.mean(gradients, dim=[0, 2, 3]) # 512
  elif method == 'gradcam++':
    b, k, u, v = gradients.size()

    alpha_num = gradients.pow(2)
    global_sum = activations.view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_denom = gradients.pow(2).mul(2) + global_sum.mul(gradients.pow(3))

    alpha = alpha_num.div(alpha_denom+1e-7)
    positive_gradients = F.relu(score.exp()*gradients)
    weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1).squeeze().detach()
    
  return weights

def get_gradcam_results(model, test_image_path, class_idx, p=90, color = [R, G, B], baseline = False, method = 'gradcam', mode = 'percentile'):
  # get the image to be tested
  test_img = process_image(test_image_path)
  img = test_img.cuda()
  img_variable = Variable(img.unsqueeze(0)).cuda()

  # activate the caching of the gradients of the output with respect to the activations when backpropagating
  pred = model(img_variable, hook=True)
  
  if class_idx is None:
    # save the predicted holonym class
    _, top_pred = torch.max(pred, dim=1)
    class_idx = top_pred.cpu().numpy()[0]
    
  score = pred[:, class_idx].squeeze()
  
  # do the back-propagation with the logit of the meronym class 
  # cache the gradients of the output with respect to the activations
  score = pred[:, class_idx] # pred has size bs x num_classes
  score.backward()

  # pull the gradients out of the model
  gradients = model.get_activations_gradient() # 1 * 512 * 14 * 14
  
  # get the activations of the last convolutional layer
  activations = model.get_activations(img[None, ...]).detach()

  weights = get_activation_weights(method, activations, gradients, score.squeeze())
  # in this way I obtain the neuron importance weights
  # these neuron weights capture the importance of each feature map k for the meronym class
  # higher positive values of the neuron importance indicate that the presence of
  # that concept leads to an increase in the class score, whereas
  # higher negative values indicate that its absence leads to an
  # increase in the score for the class.

  # weight the channels (feature maps)  by corresponding gradients
  for i in range(model.cunits):
    activations[:, i, :, :] *= weights[i]

  # average the channels (feature maps) of the activations
  heatmap = torch.mean(activations, dim=1).squeeze()

  # relu on top of the heatmap
  heatmap = np.maximum(heatmap.cpu(), 0)

  # normalize the heatmap
  heatmap = ((heatmap / torch.max(heatmap)) if (torch.count_nonzero(heatmap) != 0) else heatmap)

  # interpolate the heatmap and project it onto the original image
  img = cv2.imread(test_image_path)
  heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
  # raw normalized heatmap to be later used as input to the chosen causal metric
  raw_heatmap = deepcopy(heatmap)
  raw_threshold = np.percentile(raw_heatmap, p)
  # denormalize the heatmap
  heatmap = np.uint8(255 * heatmap)
  
  if baseline == True:
    # return the image tensor, the resized gradcam heatmap, the gradcam heatmap and the predicted label
    return img_variable, raw_heatmap, heatmap, class_idx
  
  # heatmap percentile thresholding
  if mode == 'percentile':
    threshold = np.percentile(heatmap, p)
    mask = heatmap < threshold
  elif mode == 'sum':
    # flatten the heatmap to 1D array
    flattened_heatmap = heatmap.ravel()
    # get the array indexes by value decreasing order
    decreasing_indexes = flattened_heatmap.argsort()[::-1]
    # sum all elements
    heatmap_sum = np.sum(flattened_heatmap)
    # calculate the amount of information to retain (and ablate from the image) based on the sum
    info_retain = heatmap_sum / 100 * p
    # compute a binary mask starting from all ones
    mask = np.ones_like(flattened_heatmap)
    info_acquired = 0
    for idx in decreasing_indexes:
        hm_value = flattened_heatmap[idx]
        info_acquired += hm_value
        # if I get more information than desired I stop
        if info_acquired > info_retain:
            break
        # flag the heatmap cell as a zero
        else:
            mask[idx] = 0
    mask = np.reshape(mask, (img.shape[0], img.shape[1]))
  
  img_mask = np.zeros_like(img)
  # mask over the 3 RGB channels
  img_mask[:,:,0] = mask 
  img_mask[:,:,1] = mask 
  img_mask[:,:,2] = mask
  # check if the thresolded heatmap is all zeroes
  if np.count_nonzero(img_mask) == 0:
    # no interpolation
    result_m = img
  # substitute in the original image all heatmap's zero values with the RGB chosen value
  else:
    result_m = img.copy()
    result_m[:,:,0][img_mask[:,:,0] == 0] = color[0]
    result_m[:,:,1][img_mask[:,:,1] == 0] = color[1]
    result_m[:,:,2][img_mask[:,:,2] == 0] = color[2]
  # apply color map for visualization
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  # combine image and heatmap
  superimposed_img = interpolate_img_hm(img, heatmap)
  # resize masked image to be later passed to the network
  result_m = cv2.resize(result_m, (224, 224))
  # if the thresolded heatmap is all zeroes, just take the original image
  if np.count_nonzero(img_mask) == 0:
    img_variable2 = img_variable
  else:
    cv2.imwrite('tmp_result.jpg', result_m)
    test_img2 = process_image('tmp_result.jpg')
    os.remove('tmp_result.jpg')
    # transform the masked image to Tensor
    img2 = test_img2.cuda()
    img_variable2 = Variable(img2.unsqueeze(0)).cuda()

  # return the original image Tensor, the masked image Tensor, the combined image array, the raw heatmap array, the mask array, the img path and the mask threshold
  return img_variable, img_variable2, superimposed_img, raw_heatmap, (mask if (np.count_nonzero(mask) != 0) else None), test_image_path, raw_threshold

def holonym_drop_percentage(model, image, masked_image, class_names):
  model.eval()
  # get the outputs of the network for the original image
  with torch.no_grad():
    outputs = model(image)
    outputs = F.softmax(outputs, dim=1)
    # get the outputs for the masked image as well
    outputs_m = model(masked_image)
    outputs_m = F.softmax(outputs_m, dim=1)

    # get the class index and the score for the top predicted class for the original image
    scores, preds = torch.topk(outputs, 1000)
    scores, preds = scores[0], preds[0]
    # get the top prediction label index and score
    top_class_idx = preds[0].cpu().numpy().item() # e.g., 817 for sport car
    top_class_score = scores[0].cpu().numpy().item()

    # do the same for the masked image
    m_scores, m_preds = torch.topk(outputs_m, 1000)
    m_scores, m_preds = m_scores[0], m_preds[0]
    # look for the original label index within the new prediction
    for score, pred in zip(m_scores, m_preds):
      class_idx = pred.cpu().numpy().item()
      if class_idx == top_class_idx:
        # save the score associated to the original label index when the image is perturbed
        top_class_score_m = score.cpu().numpy().item()
        break
    
    # look for the new top class (perturbed image)
    m_top_class_idx = m_preds[0].cpu().numpy().item()
    m_top_class_score = m_scores[0].cpu().numpy().item()
    # look for the new top class score in the unperturbed image prediction
    for score, pred in zip(scores, preds):
      class_idx = pred.cpu().numpy().item()
      if class_idx == m_top_class_idx:
        Y_c_top_gain = Y_c = score.cpu().numpy().item()
        O_c_top_gain = O_c = m_top_class_score
        # score gain w.r.t. the new top class
        top_gain = (np.max([0, (O_c - Y_c)]))
        break
          
    Y_c = top_class_score
    O_c = top_class_score_m

    # (max(0, Y_c - O_c) / Y_c)) * 100
    # where Y_c is the class score for class c for the original image
    # and O_c is the class score for class c for the masked image

    holonym_drop_percentage = (np.max([0, (Y_c - O_c)]) / Y_c) * 100

    return holonym_drop_percentage, top_class_idx, m_top_class_idx, top_gain, O_c_top_gain, Y_c_top_gain

"""**Generate an explanation for each meronym**"""

def create_image_explanations(test_image_path, model_name, MERONYMS, NUM_CLASSES, IDX_TO_CLASS, checkpoint, p=90, method = 'gradcam', mode = 'percentile'):
  meronym_explanations_dict = {}
  # load meronyms model
  model = load_best_model(model_name, NUM_CLASSES, save_file_name=checkpoint, freeze=False).eval()
  ckp = torch.load(checkpoint)
  per_class_f1scorec = ckp['f1score_c']

  for i, meronym in enumerate(MERONYMS):
    image, masked_image, explanation, heatmap, mask, img_path, threshold = get_gradcam_results(model, test_image_path, class_idx=i, p=p, method = method, mode = mode)
    res_heatmap = cv2.resize(np.float32(heatmap), (224, 224))
    f1scorec = per_class_f1scorec[i]
    pred_label = IDX_TO_CLASS[i]

    meronym_explanations_dict[meronym] = {'image': image, 'masked_image': masked_image, 'explanation': explanation, 'heatmap': res_heatmap, 'norm_hm': heatmap,
                                          'mask': mask, 'img_path': img_path, 'pred_label': pred_label, 'f1scorec': f1scorec, 'threshold': threshold}

  return meronym_explanations_dict

"""**Show all the masked images and the Holonym Drop %  (not shown to the end user by default)**"""

def print_tmp_img(final_img):
  cv2.imwrite('tmp.jpg', final_img) 
  image= cv2.imread('tmp.jpg')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if os.path.exists('tmp.jpg'):
    os.remove('tmp.jpg')
  plt.figure(figsize=(20,4))
  plt.axis('off')
  plt.imshow(image)
  plt.show()

def get_drop_explanations(meronym_explanations_dict, model, MERONYMS, printout=False, printbbox=False, class_names_path='./explib/class_names.json', show_gain=False):
  # Load ImageNet class labels
  with open(class_names_path) as json_file: 
    class_names = json.load(json_file)
      
  if printout == True:
    print('Debug (unfiltered) explanations')
    fig = plt.figure(figsize=(25, 6))

  meronym_drop_dict = {}
  # build an (intermediate) explanation for each meronym
  for i, meronym in enumerate(MERONYMS):
    metric = meronym_explanations_dict[meronym]['f1scorec']
    image = meronym_explanations_dict[meronym]['image']
    pred_label = meronym_explanations_dict[meronym]['pred_label']
    masked_image = meronym_explanations_dict[meronym]['masked_image']
    mask = meronym_explanations_dict[meronym]['mask']
    holonym_drop_p, pred_idx, m_idx, max_gain, O_c_max_gain, Y_c_max_gain = holonym_drop_percentage(model, image, masked_image, class_names)
    # save holonym score drop and predicted holonym class
    meronym_drop_dict[meronym] = (holonym_drop_p, class_names[str(pred_idx)].split(',')[0])
    
    if printout == True:
      ax = fig.add_subplot(1, len(MERONYMS), i+1)
      ax.axis('off')
      
      # build the title (explanation info)
      gain_str1 = (class_names[str(m_idx)].split(',')[0] + ' +' + str(np.round(max_gain, 2))) if max_gain != 0.0 else 'none'
      gain_str2 = (str(np.round(Y_c_max_gain, 5)) + ' -> ' + str(np.round(O_c_max_gain, 5)))  if max_gain != 0.0 else 'none'
      title = meronym + ' ' + 'F1-score' +': ' + str(np.round(metric, 2)) + '\n' + pred_label + '\n(' + class_names[str(pred_idx)].split(',')[0] + ' -' \
              + str(np.round(holonym_drop_p, 2)) + ' %)' + (('\n' + 'masked img label: ' + class_names[str(m_idx)].split(',')[0] + '\n' + 'gain: ' \
              + gain_str1 + '\n(' + gain_str2 + ')') if show_gain else '')
      ax.title.set_text(title)
      # retrieve the image array
      img = imshow_t(masked_image.cpu().data[0], title=meronym, ret=True)
      
      if printbbox == True and mask is not None:
          # extract the bounding boxes
          mask = cv2.cvtColor(np.float32(mask*255), cv2.COLOR_GRAY2RGB)
          mask = cv2.resize(mask, (224, 224)) # just to resize
          gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
          blurred = cv2.GaussianBlur(gray, (5, 5), 0)
          value, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
          #print_tmp_img(np.uint8(thresh))
          # Find contours
          cnts = cv2.findContours(np.uint8(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          cnts = cnts[0] if len(cnts) == 2 else cnts[1]
          #print(len(cnts))
          for c in cnts:
              x,y,w,h = cv2.boundingRect(c)
              #print(x,y,w,h)
              cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
              img = np.clip(img, 0, 1)
                
      # show the image with all the information in the title
      ax.imshow(img)
  # show the intermediate explanation(s)
  if printout == True:
    plt.show()

  return meronym_drop_dict

"""**Show each explanation only if above the Holonym Score Drop % and chosen metric thresholds**"""

def show_explanations(explanations, scores, metric_scores, titles, score_t = 10.0, metric_t = 0.7):
  color = [255,255,255] # color of the border
  font = cv2.FONT_HERSHEY_SIMPLEX
  final_img = None
  # horizontally concatenate the explanations
  for explanation, score, m_score, title in zip(explanations, scores, metric_scores, titles):
    if score > score_t and m_score > metric_t:
      if final_img is None:
        vcat = cv2.copyMakeBorder(explanation,30,10,10,10,cv2.BORDER_CONSTANT,value=color)
        cv2.putText(vcat, title, (10,25), font, 0.4, (0,0,0), 1, 0)
        final_img = vcat
      else:
        vcat = cv2.copyMakeBorder(explanation,30,10,10,10,cv2.BORDER_CONSTANT,value=color)
        cv2.putText(vcat, title, (10,25), font, 0.4, (0,0,0), 1, 0)
        final_img = cv2.hconcat((final_img, vcat))
  if final_img is None:
    print('No End-user explanation available.')
  # show the end-user explanations
  else:
    print('End-user explanations')
    cv2.imwrite('tmp.jpg', final_img) 
    image= cv2.imread('tmp.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if os.path.exists('tmp.jpg'):
      os.remove('tmp.jpg')
    plt.figure(figsize=(20,4))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def show_multiple_explanations(meronym_explanations_dict, meronym_drop_dict, MERONYMS, score_t = 10.0, metric_t = 0.7):
  explanations = []
  masks = []
  titles = []
  metric_scores = []
  for meronym in MERONYMS:
    m_score = meronym_explanations_dict[meronym]['f1scorec']
    metric_scores.append(m_score)
    explanation = meronym_explanations_dict[meronym]['explanation']
    explanations.append(explanation)
    mask = meronym_explanations_dict[meronym]['mask']
    masks.append(mask)
    title = meronym_explanations_dict[meronym]['pred_label']
    titles.append(title)
  # get the holonym score drops
  scores = [values[0] for values in meronym_drop_dict.values()]
  show_explanations(explanations, scores, metric_scores, titles, score_t = score_t, metric_t = metric_t)
   
   
"""**Show some additional explanation**"""

def show_explanation(test_image_path, model, MERONYMS, NUM_CLASSES, IDX_TO_CLASS, checkpoint, printout=False, printcomplete=False, printbbox=False, score_t=10.0, metric_t=0.7, p=90, class_names_path='./explib/class_names.json', method = 'gradcam', mode = 'percentile'):
  if printcomplete == True:
    # show original image
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    plt.figure()
    plt.axis('off')
    plt.title(test_image_path.split('/')[-1] + ' : ' + test_image_path.split('/')[-2].split(',')[0])
    plt.imshow(image)
    plt.show()
  # heatmaps creation 
  meronym_explanations_dict = create_image_explanations(test_image_path, model.name, MERONYMS, NUM_CLASSES, IDX_TO_CLASS, checkpoint, p=p, method = method, mode = mode)
  # generate explanations
  meronym_drop_dict = get_drop_explanations(meronym_explanations_dict, model, MERONYMS, printout=printout, printbbox=printbbox, class_names_path=class_names_path)
  # get the true label for the original image
  true_label = list(meronym_drop_dict.values())[0][1]
  if printcomplete == True:
    # show the end-user explanations
    show_multiple_explanations(meronym_explanations_dict, meronym_drop_dict, MERONYMS, score_t=score_t, metric_t=metric_t)

  # return holonym score drop, holonym predicted class and explanations info
  return list(meronym_drop_dict.values()), meronym_explanations_dict
  
"""**HOLMES GLOBAL HEATMAP**"""
  
def create_global_heatmap(MERONYMS, meronym_explanations_dict, drop_values):
    # collect all the meronyms' heatmaps and score drops
    heatmaps = []
    score_drops = []
    for j, meronym in enumerate(MERONYMS):
      img_tensor = meronym_explanations_dict[meronym]['image']
      sal = meronym_explanations_dict[meronym]['heatmap']
      heatmaps.append(sal)
      score_drop = drop_values[j]
      score_drops.append(score_drop)
    # normalize the score drops
    normalized_scores = list(normalize([score_drops], norm='l1')[0])
    # check the sum is 1
    if round(sum(normalized_scores), 2) != 1:
      # check if there's no drop at all
      if np.all((np.array(normalized_scores) == 0)):
        l = len(normalized_scores)
        # repartition equally the scores
        fixed_value = 1.0 / float(l)
        normalized_scores = [fixed_value] * l
      # safety check (should never happen)
      else:
        raise Exception("sum of normalized scores != 1")
    # multiply each heatmap by its associated normalized score drop
    normalized_heatmaps = [sal*score for score, sal in zip(normalized_scores, heatmaps)]
    # sum each weighted heatmap
    final_heatmap = loop_inplace_sum(normalized_heatmaps)
    
    return final_heatmap, img_tensor, heatmaps, score_drops, normalized_heatmaps, normalized_scores