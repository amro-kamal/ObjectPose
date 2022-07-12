from .vgg_original import *
from .shape_biased import *
from .simclr import *
#from .imnet_rnet import *
from .robust import *
from .taskonomy import *
from .cifar10 import *
# from .cornet import *

import torchvision
import torch 


def resnet50_untrained(pretrained=False, seed=42, **kwargs):
    torch.manual_seed(seed)
    return torchvision.models.resnet50(pretrained=False, **kwargs)