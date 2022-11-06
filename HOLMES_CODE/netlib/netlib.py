# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from netlib.mobilenet import mobilenet 
from netlib.transformers import get_transformer_model, deit_ft

"""**VGG16 with activations hooks for Grad-CAM**"""

class VGG(nn.Module):
    def __init__(self, layer_to_hook=29, cunits=512):
        super(VGG, self).__init__()
        self.name = 'vgg16'
        
        # get the pretrained VGG16 network
        self.vgg = torchvision.models.vgg16(pretrained=True)
        
        # dissect the network to access its last convolutional layer
        self.features = self.vgg.features[:(layer_to_hook+1)]
        
        self.cunits = cunits
        
        # get the max pool of the features
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg16
        self.classifier = self.vgg.classifier

        delattr(self, 'vgg')
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=False):
        x = self.features(x)
        
        if hook == True:
          # register the hook
          # the hook will be called every time a gradient with respect to the tensor is computed
          h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation extraction
    def get_activations(self, x):
        return self.features(x)
        
"""**VGG16 Meronyms Model**"""

def vgg16_ft(NUM_CLASSES, edit=True, freeze=True):
    model = VGG()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier (unfrozen weights)
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, NUM_CLASSES),
        )

    return model
    
"""**ResNet50 with activations hooks for Grad-CAM**"""

class ResNet(nn.Module):
    def __init__(self, cunits = (4 * 512)):
        super(ResNet, self).__init__()
        self.name = 'resnet50'
        
        # get the pretrained ResNet50 network
        self.resnet = torchvision.models.resnet50(pretrained=True)
        
        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        
        # get the average pool of the features
        self.avgpool = self.resnet.avgpool
        
        # get the classifier of the resnet50
        self.classifier = self.resnet.fc
        
        self.cunits = cunits

        delattr(self, 'resnet')
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=False):
        x = self.features(x)
        
        if hook == True:
          # register the hook
          # the hook will be called every time a gradient with respect to the tensor is computed
          h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation extraction
    def get_activations(self, x):
        return self.features(x)
        
"""**ResNet50 Meronyms Model**"""

def resnet50_ft(NUM_CLASSES, edit=True, freeze=True):
    model = ResNet()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier (unfrozen weights)
    block_expansion = 4
    model.classifier = nn.Linear(512 * block_expansion, NUM_CLASSES)

    return model
  
"""**DenseNet121 with activations hooks for Grad-CAM**"""

class DenseNet(nn.Module):
    def __init__(self, cunits = 1024):
        super(DenseNet, self).__init__()
        self.name = 'densenet121'
        
        # get the pretrained DenseNet121 network
        self.densenet = torchvision.models.densenet121(pretrained=True)
        
        # dissect the network to access its last convolutional layer
        self.features = self.densenet.features
        
        # get the classifier of the densenet121
        self.classifier = self.densenet.classifier
        
        self.cunits = cunits

        delattr(self, 'densenet')
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=False):
        x = self.features(x)
        
        if hook == True:
          # register the hook
          # the hook will be called every time a gradient with respect to the tensor is computed
          h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation extraction
    def get_activations(self, x):
        return self.features(x)
        
"""**DenseNet121 Meronyms Model**"""

def densenet121_ft(NUM_CLASSES, edit=True, freeze=True):
    model = DenseNet()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier (unfrozen weights)
    model.classifier = nn.Linear(1024, NUM_CLASSES)

    return model
    
"""**Mobilenet with activations hooks for Grad-CAM**"""

class MobileNet(nn.Module):
    def __init__(self, cunits=1024):
        super(MobileNet, self).__init__()
        
        self.name = 'mobilenet'
        
        # get the pretrained mobilenet network
        self.mobilenet = mobilenet(pretrained=True)
        
        # dissect the network to access its last convolutional layer
        self.features = self.mobilenet.features
        
        self.cunits = cunits
        
        # get the avg pool of the features
        self.avgpool = self.mobilenet.avgpool
        
        # get the classifier of mobilenet
        self.classifier = self.mobilenet.fc

        delattr(self, 'mobilenet')
        
        # placeholder for the gradients
        self.gradients = None
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=False):
        x = self.features(x)
        
        if hook == True:
          # register the hook
          # the hook will be called every time a gradient with respect to the tensor is computed
          h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x
    
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation extraction
    def get_activations(self, x):
        return self.features(x)
        
"""**MobileNet Meronyms Model**"""

def mobilenet_ft(NUM_CLASSES, edit=True, freeze=True):
    model = MobileNet()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier (unfrozen weights)
    model.classifier = nn.Linear(1024, NUM_CLASSES)

    return model
    
"""**Inception with activations hooks for Grad-CAM**"""

class Inception(nn.Module):
    def __init__(self, cunits=1024):
        super(Inception, self).__init__()
        
        self.name = 'inception'
        
        # get the pretrained inception network
        self.inception = torchvision.models.googlenet(pretrained=True)
        
        self.features = nn.Sequential(self.inception.conv1,
                                      self.inception.maxpool1,
                                      self.inception.conv2,
                                      self.inception.conv3,
                                      self.inception.maxpool2,
                                      self.inception.inception3a,
                                      self.inception.inception3b,
                                      self.inception.maxpool3,
                                      self.inception.inception4a,
                                      self.inception.inception4b,
                                      self.inception.inception4c,
                                      self.inception.inception4d,
                                      self.inception.inception4e,
                                      self.inception.maxpool4,
                                      self.inception.inception5a,
                                      self.inception.inception5b)                                      
        
        self.cunits = cunits
        
        self.avgpool = self.inception.avgpool
        self.dropout = self.inception.dropout
        
        # get the classifier of inception
        self.classifier = self.inception.fc

        delattr(self, 'inception')
        
        # placeholder for the gradients
        self.gradients = None
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=False):
        x = self.features(x)
        
        if hook == True:
          # register the hook
          # the hook will be called every time a gradient with respect to the tensor is computed
          h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation extraction
    def get_activations(self, x):
        return self.features(x)
        
"""**Inception Meronyms Model**"""

def inception_ft(NUM_CLASSES, edit=True, freeze=True):
    model = Inception()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier (unfrozen weights)
    model.classifier = nn.Linear(1024, NUM_CLASSES)

    return model
    
###################################################
    
def load_best_model(model_name, NUM_CLASSES, save_file_name, freeze=True):
  # Load the best (classifier) state dict
  checkpoint = torch.load(save_file_name)
  if model_name == 'vgg16':
    model = vgg16_ft(NUM_CLASSES, freeze=freeze)
  elif model_name == 'resnet50':
    model = resnet50_ft(NUM_CLASSES, freeze=freeze)
  elif model_name == 'densenet121':
    model = densenet121_ft(NUM_CLASSES, freeze=freeze)
  elif model_name == 'mobilenet':
    model = mobilenet_ft(NUM_CLASSES, freeze=freeze)
  elif model_name == 'inception':
    model = inception_ft(NUM_CLASSES, freeze=freeze)
  elif model_name == 'deit':
    model = deit_ft(NUM_CLASSES, freeze=freeze)
  else:
    raise Exception('Model {} not supported.'.format(model_name))
  model.classifier.load_state_dict(checkpoint['model_state_dict'])

  return model.cuda()

"""**Load the original models**"""

def get_vgg_model():
  vgg_model = models.vgg16(pretrained=True).cuda()
  vgg_model.name = 'vgg16'
  vgg_model.eval()

  return vgg_model
  
def get_resnet_model():
  resnet_model = models.resnet50(pretrained=True).cuda()
  resnet_model.name = 'resnet50'
  resnet_model.eval()

  return resnet_model
  
def get_densenet_model():
  densenet_model = models.densenet121(pretrained=True).cuda()
  densenet_model.name = 'densenet121'
  densenet_model.eval()

  return densenet_model
  
def get_mobilenet_model():
  mobilenet_model = mobilenet(pretrained=True).cuda()
  mobilenet_model.name = 'mobilenet'
  mobilenet_model.eval()

  return mobilenet_model
  
def get_inception_model():
  inception_model = models.googlenet(pretrained=True).cuda()
  inception_model.name = 'inception'
  inception_model.eval()

  return inception_model
  
def get_model_by_name(model_name, hooks=False):
    if model_name == 'vgg16':
      model = get_vgg_model() if hooks == False else vgg16_ft(None, edit=False).cuda().eval()
    elif model_name == 'mobilenet':
      model = get_mobilenet_model() if hooks == False else mobilenet_ft(None, edit=False).cuda().eval()
    elif model_name == 'inception':
      model = get_inception_model() if hooks == False else inception_ft(None, edit=False).cuda().eval()
    elif model_name == 'deit':
      model = get_transformer_model() if hooks == False else deit_ft(None, edit=False).cuda().eval()
    else:
      raise Exception('Model {} not supported.'.format(model_name))
      
    return model