import torch
from torch.hub import load_state_dict_from_url
from visualpriors.taskonomy_network import TaskonomyEncoder, LIST_OF_TASKS
from visualpriors.transforms import TASKONOMY_PRETRAINED_URLS


encoders = ['_'.join([task, 'encoder']) for task in LIST_OF_TASKS] # names of taskonomy encoder networks

__all__ = encoders

def _model(name, pretrained, progress, use_data_parallel, **kwargs):
    model = TaskonomyEncoder()
    if pretrained:
        checkpoint = load_state_dict_from_url(TASKONOMY_PRETRAINED_URLS[name], progress=progress)
        model.load_state_dict(checkpoint['state_dict'])    
    model = torch.nn.DataParallel(model) if use_data_parallel else model 
    return model


def autoencoding_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """Autoencoding encoder network"""
    return _model("autoencoding_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)
    

def class_object_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("class_object_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def class_scene_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("class_scene_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def colorization_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("colorization_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def curvature_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("curvature_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def denoising_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("denoising_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def depth_euclidean_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("depth_euclidean_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def depth_zbuffer_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("depth_zbuffer_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def edge_occlusion_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("edge_occlusion_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def edge_texture_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("edge_texture_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def egomotion_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("egomotion_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def fixated_pose_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("fixated_pose_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def inpainting_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("inpainting_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def jigsaw_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("jigsaw_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def keypoints2d_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("keypoints2d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def keypoints3d_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("keypoints3d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def nonfixated_pose_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("nonfixated_pose_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def normal_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("normal_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def point_matching_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("point_matching_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def reshading_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("reshading_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def room_layout_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("room_layout_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def segment_semantic_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("segment_semantic_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def segment_unsup25d_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("segment_unsup25d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def segment_unsup2d_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("segment_unsup2d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)


def vanishing_point_encoder(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    """ encoder network"""
    return _model("vanishing_point_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel,  **kwargs)