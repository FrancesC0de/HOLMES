import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Callable, Any

class MobileNetV1(nn.Module):
    def __init__(self, ch_in: int = 3, n_classes: int = 1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.features = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
        
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
    
def substitute_prefix(text, prefix, new_prefix):
    if text.startswith(prefix):
        text = new_prefix + text[len(prefix):]
    return text
        
def mobilenet(pretrained: bool = False, **kwargs: Any) -> MobileNetV1:
    model = MobileNetV1(**kwargs)
    if pretrained:
        try:
            state_dict = torch.load("netlib/mobilenet.pth")['state_dict']
        except:
            state_dict = torch.load("../netlib/mobilenet.pth")['state_dict']
        # rename dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            # remove "module." prefix
            new_key = remove_prefix(key, "module.")
            # substitute "model" with "features"
            new_key = substitute_prefix(new_key, "model", "features")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    return model
    