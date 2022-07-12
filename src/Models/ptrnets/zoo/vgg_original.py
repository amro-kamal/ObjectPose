import torch
from torch import nn
import os
from torchvision.models import vgg19, vgg16
from ..utils.gdrive import load_state_dict_from_google_drive
from torch.hub import load_state_dict_from_url

__all__ = ['vgg19_original', 'vgg19_norm']


google_drive_ids = {
    'vgg19_original': '18KRngGJMAhQJmlzjHmgyXuNjqd2l6rQG',
    'vgg19_norm'    : '1r2MAofFyBy3TyazQ7NQOpoAr1dDEAKzL',
}

model_urls = {
    'vgg19_original': '',
    'vgg19_norm'    : '',
}    
    

# Define vgg auxiliary class
class VGGConv(nn.Module):
    def __init__(self):
        super().__init__()
        _vgg19m = vgg19()
        self.features = _vgg19m.features
        self.avgpool  = _vgg19m.avgpool
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def _vgg19conv(arch, pretrained, progress, **kwargs):
    model = VGGConv()
    if pretrained:
        try:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        except:
            state_dict = load_state_dict_from_google_drive(google_drive_ids[arch],
                                                  progress=progress, filename = '{}_conv.pth'.format(arch))
        model.features.load_state_dict(state_dict)
    return model
    

def vgg19_original(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>` ONLY CONVOLUTIONAL LAYERS
    The weights of this network are those of the original publication and not those from the network trained in pytorch
    Args:
        pretrained (bool): If True, returns the convlayers pre-trained on ImageNet (original weights)
    """
    return _vgg19conv('vgg19_original', pretrained, progress, **kwargs)
    

def vgg19_norm(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>` ONLY CONVOLUTIONAL LAYERS 
    The weights of this network are those of the original publication normalized so that outputs have mean and std equal to one over ImageNet
    Args:
        pretrained (bool): If True, returns the conv layers pre-trained on ImageNet (original weights)
    """
    return _vgg19conv('vgg19_norm', pretrained, progress, **kwargs)



from collections import OrderedDict

RF_SIZES = OrderedDict()
RF_SIZES['conv1/conv1_1'] = 3
RF_SIZES['conv1/conv1_2'] = 5
RF_SIZES['pool1'] = 6
RF_SIZES['conv2/conv2_1'] = 10
RF_SIZES['conv2/conv2_2'] = 14
RF_SIZES['pool2'] = 16
RF_SIZES['conv3/conv3_1'] = 24
RF_SIZES['conv3/conv3_2'] = 32
RF_SIZES['conv3/conv3_3'] = 40
RF_SIZES['conv3/conv3_4'] = 48
RF_SIZES['pool3'] = 52
RF_SIZES['conv4/conv4_1'] = 68
RF_SIZES['conv4/conv4_2'] = 84
RF_SIZES['conv4/conv4_3'] = 100
RF_SIZES['conv4/conv4_4'] = 116
RF_SIZES['pool4'] = 124
RF_SIZES['conv5/conv5_1'] = 156
RF_SIZES['conv5/conv5_2'] = 188
RF_SIZES['conv5/conv5_3'] = 220
RF_SIZES['conv5/conv5_4'] = 252
RF_SIZES['pool5'] = 268

LAYER_ID = OrderedDict()
LAYER_ID['conv1/conv1_1'] = 1
LAYER_ID['conv1/conv1_2'] = 3
LAYER_ID['pool1'] = 5
LAYER_ID['conv2/conv2_1'] = 6
LAYER_ID['conv2/conv2_2'] = 8
LAYER_ID['pool2'] = 10
LAYER_ID['conv3/conv3_1'] = 11
LAYER_ID['conv3/conv3_2'] = 13
LAYER_ID['conv3/conv3_3'] = 15
LAYER_ID['conv3/conv3_4'] = 17
LAYER_ID['pool3'] = 19
LAYER_ID['conv4/conv4_1'] = 20
LAYER_ID['conv4/conv4_2'] = 22
LAYER_ID['conv4/conv4_3'] = 24
LAYER_ID['conv4/conv4_4'] = 26
LAYER_ID['pool4'] = 28
LAYER_ID['conv5/conv5_1'] = 29
LAYER_ID['conv5/conv5_2'] = 31
LAYER_ID['conv5/conv5_3'] = 33
LAYER_ID['conv5/conv5_4'] = 35
LAYER_ID['pool5'] = 37


NUM_FEATMAPS = OrderedDict()
fmap_groups = [64,128,256,512,512]
for k in RF_SIZES.keys():
    NUM_FEATMAPS[k] = fmap_groups[int(k.split('/')[0][-1]) - 1]
