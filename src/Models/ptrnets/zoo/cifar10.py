import os
import torch
from torchvision.models import resnet50
from ..utils.gdrive import load_state_dict_from_google_drive
from torch.hub import load_state_dict_from_url

__all__ = ['resnet50_cifar10', 'resnet50_cifar10_corrupt0', 'resnet50_cifar10_corrupt0_2', 'resnet50_cifar10_corrupt0_4',\
           'resnet50_cifar10_corrupt0_6', 'resnet50_cifar10_corrupt0_8', 'resnet50_cifar10_corrupt1']

google_drive_ids = {
    'resnet50_cifar10': '1sS1U0y-EKXBAxq1-bxkOqmnfJtRk8z2A',
    'resnet50_cifar10_corrupt0': '1ihR9o1xzb5BggBdvRRwt76g9vrv5sBRF',
    'resnet50_cifar10_corrupt0_2': '1fDF8bKOIXPQ2yReTcQPk3ABfAiGtwbKR',
    'resnet50_cifar10_corrupt0_4': '1NbY3BWySr4zMXuRnbjlUz2uMBqfisoAH',
    'resnet50_cifar10_corrupt0_6': '11DS-ZAfXrCRtFe-1EgF8khpwN7flVTh_',
    'resnet50_cifar10_corrupt0_8': '1wb5RMVOjP3a-5tPImgEzlzVx4AWuUWWO',
    'resnet50_cifar10_corrupt1': '14Bdfw3rr0LP0Y9Ct7pVtseYPxRdwsvdk',
}


model_urls = {}

def _model(arch, model_fn, pretrained, progress, use_data_parallel, **kwargs):
    
    model = model_fn(pretrained=False, num_classes=10)   
    
    model = torch.nn.DataParallel(model) if use_data_parallel else model
        
    if pretrained:
        try:
            checkpoint = load_state_dict_from_url(model_urls[arch], progress=progress)
        except:
            checkpoint = load_state_dict_from_google_drive(google_drive_ids[arch],
                                                  progress=progress, filename = '{}.pth.tar'.format(arch), **kwargs)
        
        if use_data_parallel:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            new_dict = dict()
            for k, v in checkpoint['state_dict'].items():
                new_dict.update({k.replace("module.", ""): v})                
            model.load_state_dict(new_dict)
        
    return model


def resnet50_cifar10(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 starting with seed=42, and with data augmentation
    """
    return _model('resnet50_cifar10', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt0(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 starting with seed=42, and without data augmentation
    """
    return _model('resnet50_cifar10_corrupt0', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt0_2(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 with label corruption prob = 0.2, starting with seed=42, and without data augmentation
    """
    return _model('resnet50_cifar10_corrupt0_2', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt0_4(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 with label corruption prob = 0.4, starting with seed=42, and without data augmentation
    """
    return _model('resnet50_cifar10_corrupt0_4', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt0_6(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 with label corruption prob = 0.6, starting with seed=42, and without data augmentation
    """
    return _model('resnet50_cifar10_corrupt0_6', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt0_8(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 with label corruption prob = 0.8, starting with seed=42, and without data augmentation
    """
    return _model('resnet50_cifar10_corrupt0_8', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt1(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 trained on Cifar10 with label corruption prob = 1.0, starting with seed=42, and without data augmentation
    """
    return _model('resnet50_cifar10_corrupt1', resnet50, pretrained, progress, use_data_parallel, **kwargs)