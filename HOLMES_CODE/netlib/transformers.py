import torch
import torch.nn as nn
import numpy as np


# https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
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


"""**Tranformer with activations hooks for Grad-CAM**"""

# https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__()
        self.name = 'deit'
        
        # get the pretrained Transformer network
        self.deit = torch.hub.load('facebookresearch/deit:main',
                            'deit_tiny_patch16_224', pretrained=True)

        # inherit attributes and objects
        self.num_classes = self.deit.num_classes
        self.global_pool = self.deit.global_pool
        self.num_features = self.deit.embed_dim
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
        self.head = self.deit.head
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def classifier(self, x):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return self.head(x)

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
        x = self.classifier(x)

        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return reshape_transform(self.gradients)
    
    # method for the activation extraction
    def get_activations(self, x):
        return reshape_transform(self.features(x))


"""**Transformer Meronyms Model**"""

def tranformer_ft(NUM_CLASSES, edit=True, freeze=True):
    model = Transformer()
    if edit == False:
      return model
    if freeze == True:
      # freeze model weights
      for param in model.parameters():
          param.requires_grad = False
    # substitute the classifier head (unfrozen weights)
    model.head = nn.Linear(model.embed_dim, NUM_CLASSES) 

    return model


"""**Load the original model**"""

def get_transformer_model():
  vit_model = torch.hub.load('facebookresearch/deit:main',
                            'deit_tiny_patch16_224', pretrained=True).cuda()
  vit_model.name = 'deit'
  vit_model.eval()

  return vit_model


def main():
    # check that the custom transformer model is equivalent to the original model
    batch_size = 8
    inp = torch.rand(batch_size, 3, 224, 224).cuda()
    model_orig = get_transformer_model()
    model_custom = tranformer_ft(1000, edit=False).eval().cuda()

    with torch.no_grad():
        out_orig = model_orig(inp).cpu().detach().numpy()
        out_custom = model_custom(inp).cpu().detach().numpy()

    np.testing.assert_array_equal(out_orig, out_custom)

    print("PASSED: models are equivalent.")


if __name__ == "__main__":
    main()