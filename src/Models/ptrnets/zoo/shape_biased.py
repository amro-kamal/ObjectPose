import os
import torch
from torchvision.models import vgg16, resnet50, alexnet
from ..utils.gdrive import load_state_dict_from_google_drive
from torch.hub import load_state_dict_from_url

__all__ = ['resnet50_trained_on_SIN', 'resnet50_trained_on_SIN_and_IN', 'vgg16_trained_on_SIN', 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN', 'alexnet_trained_on_SIN']


google_drive_ids = {
    'resnet50_trained_on_SIN': '',
    'resnet50_trained_on_SIN_and_IN': '',
    'vgg16_trained_on_SIN': '1PMZCo2ASbilrHoqmDO7XZkQWkA0cxoDQ',
    'alexnet_trained_on_SIN': '',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN':'',
}


model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'vgg16_trained_on_SIN': '',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
}



def _model(arch, model_fn, pretrained, progress, use_data_parallel, **kwargs):
    
    model = model_fn(pretrained=False)   
    if 'vgg' in arch:
        model.features = torch.nn.DataParallel(model.features) if use_data_parallel else model.features
    else:
        model = torch.nn.DataParallel(model) if use_data_parallel else model
        
    
    if pretrained:
        try:
            checkpoint = load_state_dict_from_url(model_urls[arch], progress=progress)
        except:
            checkpoint = load_state_dict_from_google_drive(google_drive_ids[arch],
                                                  progress=progress, filename = '{}.pth'.format(arch), **kwargs)
        
        if use_data_parallel:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            new_dict = dict()
            for k, v in checkpoint['state_dict'].items():
                new_dict.update({k.replace("module.", ""): v})                
            model.load_state_dict(new_dict)
        
    return model


def resnet50_trained_on_SIN(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Sylized ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy 
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX   
    """
    return _model('resnet50_trained_on_SIN', resnet50, pretrained, progress, use_data_parallel, **kwargs)
    

def resnet50_trained_on_SIN_and_IN(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Sylized ImageNet + standard ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy 
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX   
    """
    return _model('resnet50_trained_on_SIN_and_IN', resnet50, pretrained, progress, use_data_parallel, **kwargs)

 
def resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Sylized ImageNet + standard ImageNet. Then finetuned on ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy 
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX   
    """
    return _model('resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def vgg16_trained_on_SIN(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Vgg16 trained on Sylized ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy 
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX   
    """
    return _model('vgg16_trained_on_SIN', vgg16, pretrained, progress, use_data_parallel, **kwargs)


def alexnet_trained_on_SIN(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" AlexNet trained on Sylized ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy 
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX   
    """
    return _model('alexnet_trained_on_SIN', alexnet, pretrained, progress, use_data_parallel, **kwargs)
