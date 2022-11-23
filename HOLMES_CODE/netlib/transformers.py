import torch
import torch.nn as nn
import numpy as np
import timm
from itertools import chain

# https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class BlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, x_norm1):
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(x_norm1)))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))
        return x

class SwinBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, x_norm1=None):
        H, W = self.block.input_resolution
        B, L, C = x.shape
        if not L == H * W:
            sys.exit(-156)

        shortcut = x
        if x_norm1 is None: # get output up to x_norm2 (excluded)
            n2 = True
            x = self.block.norm1(x)
        else: # get full output
            n2 = False
            x = x_norm1
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.block.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.block.shift_size, -self.block.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.block.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.block.window_size * self.block.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.block.attn(x_windows, mask=self.block.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.block.window_size, self.block.window_size, C)
        shifted_x = window_reverse(attn_windows, self.block.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.block.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.block.shift_size, self.block.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.block.drop_path(x)
        if n2: return x
        x = x + self.block.drop_path(self.block.mlp(self.block.norm2(x)))
        return x


class SwinBlockMLPDrop(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.drop_path = block.drop_path
        self.mlp = block.mlp

    def forward(self, x, x_norm2):
        x = x + self.drop_path(self.mlp(x_norm2))
        return x


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


MODEL_ZOO = {
    "deit-tiny": {
        "repo" : "facebookresearch/deit:main",
        "model": "deit_tiny_patch16_224"
    },
    "deit-base": {
        "repo" : "facebookresearch/deit:main",
        "model": "deit_base_patch16_224"
    }
}


"""**Tranformer with activations hooks for Grad-CAM**"""

# https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class Transformer(nn.Module):
    def __init__(self, model_name="deit-base", *args, **kwargs):
        super(Transformer, self).__init__()
        self.name = 'deit'
        
        # get the pretrained Transformer network
        model_dict = MODEL_ZOO.get(model_name)
        self.deit = torch.hub.load(model_dict["repo"],
                            model_dict["model"], pretrained=True)

        # inherit attributes and objects
        self.num_classes = self.deit.num_classes
        self.global_pool = self.deit.global_pool
        self.cunits = self.num_features = self.embed_dim = self.deit.embed_dim
        self.num_prefix_tokens = self.deit.num_prefix_tokens
        self.no_embed_class = self.deit.no_embed_class
        self.patch_embed = self.deit.patch_embed
        # objects
        self.cls_token = self.deit.cls_token
        self.pos_embed = self.deit.pos_embed
        self.pos_drop = self.deit.pos_drop
        self.norm_pre = self.deit.norm_pre
        # function
        self._pos_embed = self.deit._pos_embed
        
        # dissect the network to access its last norm layer
        self.features = self.deit.blocks[:-1]
        # N.B: we can chose any layer before the final attention block
        # the gradient will be zero afterwards
        # last eligible layer is at index o (norm1) of last block (-1)
        self.target_layer = self.deit.blocks[-1].norm1

        # add the remainig blocks
        self.features2 = BlockWrapper(self.deit.blocks[-1])
        
        # get the classifier head of the transformer
        self.norm = self.deit.norm
        self.fc_norm = self.deit.fc_norm
        self.classifier = self.deit.head
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, hook=False):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.features(x)
        x_n = self.target_layer(x)

        if hook == True:
          # register the hook
          # the hook will be called every time a gradient with respect to the tensor is computed
          h = x_n.register_hook(self.activations_hook)

        x = self.features2(x, x_n)
        x = self.norm(x)

        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.classifier(x)

        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return reshape_transform(self.gradients)
    
    # method for the activation extraction
    def get_activations(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.features(x)
        x = self.target_layer(x)
        return reshape_transform(x)


"""**Transformer Meronyms Model**"""

def deit_ft(NUM_CLASSES, edit=True, freeze=True, deep_classifier=False):
    model = Transformer()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier head (unfrozen weights)
    if not deep_classifier:
        model.classifier = nn.Linear(model.embed_dim, NUM_CLASSES) 
    else:
        hidden_size = model.embed_dim // 2
        model.classifier = nn.Sequential(
            nn.Linear(model.embed_dim, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, NUM_CLASSES),
        )

    return model


class Swin(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swin, self).__init__()
        self.name = 'swin'
        self.target = "f3_n1"

        # get the pretrained Transformer network
        self.swin = timm.create_model('swin_small_patch4_window7_224', pretrained=True)

        # inherit attributes and objects
        self.num_classes = self.swin.num_classes
        self.global_pool = self.swin.global_pool
        self.cunits = self.num_features = self.swin.num_features
        self.embed_dim = self.swin.embed_dim
        #self.num_prefix_tokens = self.swin.num_prefix_tokens
        #self.no_embed_class = self.swin.no_embed_class
        self.patch_embed = self.swin.patch_embed
        # objects
        #self.cls_token = self.swin.cls_token
        self.pos_embed = self.swin.absolute_pos_embed
        self.pos_drop = self.swin.pos_drop
        #self.norm_pre = self.swin.norm_pre
        # function
        #self._pos_embed = self.swin._pos_embed

        # dissect the network to access its last norm layer
        self.features = self.swin.layers[:-1]
        self.features2 = self.swin.layers[-1].blocks[:-1]
        # N.B: we can choose any layer before the final attention block
        # the gradient will be zero afterwards
        # last eligible layer is at index o (norm1) of last block (-1)swin

        if self.target == "f3_n1":
            self.target_layer = self.swin.layers[-1].blocks[-1].norm1
            self.features3 = SwinBlockWrapper(self.swin.layers[-1].blocks[-1])
        elif self.target == "f2_n1":
            self.target_layer = self.swin.layers[-1].blocks[-2].norm1
            self.features2 = SwinBlockWrapper(self.swin.layers[-1].blocks[-2])
            self.features3 = self.swin.layers[-1].blocks[-1]
        elif self.target == "f2_n2":
            self.features2 = SwinBlockWrapper(self.swin.layers[-1].blocks[-2])
            self.target_layer = self.swin.layers[-1].blocks[-2].norm2
            self.features2_bis = SwinBlockMLPDrop(self.swin.layers[-1].blocks[-2])
            self.features3 = self.swin.layers[-1].blocks[-1]

        # get the classifier head of the transformer
        self.norm = self.swin.norm
        #self.fc_norm = self.swin.fc_norm
        self.classifier = self.swin.head

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, hook=False):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.features(x)
        #x = self._pos_embed(x)
        #x = self.norm_pre(x)
        if self.target == "f3_n1":
            x = self.features2(x)
            x_n = self.target_layer(x)
            if hook == True: h = x_n.register_hook(self.activations_hook)
            x = self.features3(x, x_n)
        elif self.target == "f2_n1":
            x_n = self.target_layer(x)
            if hook == True: h = x_n.register_hook(self.activations_hook)
            x = self.features2(x, x_n)
            x = self.features3(x)
        elif self.target == "f2_n2":
            x = self.features2(x)
            x_n2 = self.target_layer(x)
            if hook == True: h = x_n2.register_hook(self.activations_hook)
            x = self.features2_bis(x, x_n2)
            x = self.features3(x)

        x = self.norm(x)

        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        #x = self.fc_norm(x)
        x = self.classifier(x)

        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return swin_reshape_transform(self.gradients)

    # method for the activation extraction
    def get_activations(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.features(x)
        if self.target == "f3_n1":
            x = self.features2(x)
            x = self.target_layer(x)
        elif self.target == "f2_n1":
            x = self.target_layer(x)
        elif self.target == "f2_n2":
            x = self.features2(x)
            x = self.target_layer(x)
        return swin_reshape_transform(x)


"""**Transformer Meronyms Model**"""


def swin(NUM_CLASSES, edit=True, freeze=True):
    model = Swin()
    if edit == False:
        return model
    if freeze == True:
        # freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        for param in chain(model.features3.parameters(), model.target_layer.parameters()):
            param.requires_grad = True
    # substitute the classifier head (unfrozen weights)
    model.classifier = nn.Linear(model.num_features, NUM_CLASSES)

    return model


"""**Load the original model**"""

def get_transformer_model(model_name="deit-base"):
  model_dict = MODEL_ZOO.get(model_name)
  vit_model = torch.hub.load(model_dict["repo"],
                    model_dict["model"], pretrained=True).cuda()
  vit_model.name = 'deit'
  vit_model.eval()

  return vit_model

def get_swin_model():
  vit_model = timm.create_model('swin_small_patch4_window7_224', pretrained=True).cuda()
  vit_model.name = 'swin'
  vit_model.eval()

  return vit_model

def main():
    # check that the custom transformer model is equivalent to the original model
    batch_size = 8
    inp = torch.rand(batch_size, 3, 224, 224).cuda()
    model_orig = get_swin_model() #get_transformer_model()
    model_custom = swin(1000, edit=False).eval().cuda() #deit_ft(1000, edit=False).eval().cuda()

    with torch.no_grad():
        out_orig = model_orig(inp).cpu().detach().numpy()
        out_custom = model_custom(inp).cpu().detach().numpy()

    np.testing.assert_array_equal(out_orig, out_custom)

    print("PASSED: models are equivalent.")


if __name__ == "__main__":
    main()