from .CLIP import clip
from .ptrnets import simclr_resnet50x1
from pytorch_pretrained_vit import ViT
import timm
from transformers import BeitFeatureExtractor, BeitForImageClassification
import torch
import torchvision
from utils import imagenet_classes_for_clip, imagenet_pose_templates, imagenet_templates, imagenet_single_template
from torchvision import transforms
import math
from PIL import Image


#A general Model class
class Model:
    def __init__(self, model, device, transform=None, in_mean=[0.485, 0.456, 0.406], in_std=[0.229, 0.224, 0.225], input_size=224, crop_pct=0.875, interpolation=Image.BILINEAR):
        # assert model==None, "You have to sepecify the model"
        '''
            model: Pytorch model
            device: CPU | GPU
            transform: torchvision transforms.
            in_mean: mean ImageNet RGB values for preprocessing normalization.
            in_std: std for ImageNet RGB values for preprocessing normalization.
            input_size: input image size (the value depends on the model).
            crop_pct: cropping ratio for preprocessing (the value depends on the model).
            interpolation: interpolation for the Resize() function (BILINEAR | BICUBIC).
        
        '''
        self.in_mean = in_mean
        self.in_std = in_std
        self.model = model.to(device) 
        self.model.eval()
        self.input_size = input_size
        self.crop_pct = crop_pct
        self.interpolation = interpolation
        self.transform = transform #will be "None" at the begining for most of the models 

   
    def get_transforms(self):
        return self.transform
    
    def set_transforms(self):
        if self.transform==None:
            scale_size = int(math.floor(self.input_size / self.crop_pct))
            self.transform = transforms.Compose( [ 
                                              
                transforms.Resize(scale_size, interpolation=self.interpolation),
                transforms.CenterCrop(size=(self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.in_mean, std=self.in_std)
                ])


    def __call__(self, batch):
        '''
           Function to call the model
           return:
               logits
        '''
        
        with torch.no_grad():
            output = self.model(batch)
            # probs = output.softmax(dim=-1)

        return output


class ClipEnsemble:
    '''
    CLIP preprocessing:
        Compose(
        Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
        CenterCrop(size=(224, 224))
        <function _convert_image_to_rgb at 0x7f26a9a16a70>
        ToTensor()
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )
    '''

    def __init__(self, model_name, imagenet_classes, imagenet_templates, device, *args):
        self.model, clip_transforms = clip.load(model_name, device=device)
        self.imagenet_classes = imagenet_classes
        self.imagenet_templates = imagenet_templates
        self.device = device
        self.avg_text_features = None
        self.transform= clip_transforms
        self.crp_pct = 1.0
        
    def _get_avg_text_representaion(self, class_names, templates):
        '''
        function to get the text representation from CLIP text encoder
        '''
        with torch.no_grad():
            zeroshot_avg_text_features = []
            #run the model for each class alone (each class with the 80 prompts)
            for class_name in class_names:
                texts = [template.format(class_name) for template in templates]  # format with class, should be list
                
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                avg_class_embedding = class_embeddings.mean(dim=0) #the mean over the templates
                avg_class_embedding /= avg_class_embedding.norm()
                zeroshot_avg_text_features.append(avg_class_embedding)
            zeroshot_avg_text_features = torch.stack(zeroshot_avg_text_features, dim=1).to(self.device)

        return zeroshot_avg_text_features

    def get_transforms(self):
        return self.clip_transforms

    def set_transforms(self):
        pass
        
    def __call__(self, images):
        assert type(images) is torch.Tensor, 'the batch input to CLIP should be tensor'

        self.model.eval()
        if self.avg_text_features == None:
          self.avg_text_features = self._get_avg_text_representaion(self.imagenet_classes, self. imagenet_templates)

        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = 100. * image_features @ self.avg_text_features
        return similarity




Resnet50_model = lambda device:  Model(model=torchvision.models.resnet50(pretrained=True), device=device)
Resnet152_model = lambda device:  Model(model=torchvision.models.resnet152(pretrained=True), device=device)
Resnet101_model = lambda device:  Model(model=torchvision.models.resnet101(pretrained=True), device=device)

#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
ConvNext_XL_model = lambda device: Model(model=timm.create_model('convnext_xlarge_in22ft1k', pretrained=True), interpolation=Image.BICUBIC, device=device)
ConvNext_L_384_model = lambda device: Model(model=timm.create_model('convnext_large_384_in22ft1k', pretrained=True), input_size=384, crop_pct=1.0, interpolation=Image.BICUBIC, device=device)
ConvNext_L_model = lambda device: Model(model=timm.create_model('convnext_large_in22ft1k', pretrained=True), interpolation=Image.BICUBIC, device=device)
ConvNext_B_384_model= lambda device: Model(model=timm.create_model('convnext_base_384_in22ft1k', pretrained=True), input_size=384, crop_pct=1.0, interpolation=Image.BICUBIC, device=device)
ConvNext_B_model = lambda device: Model(model=timm.create_model('convnext_base_in22ft1k', pretrained=True), interpolation=Image.BICUBIC, device=device)
ConvNext_L_in_model = lambda device: Model(model=timm.create_model('convnext_large', pretrained=True), interpolation=Image.BICUBIC, device=device)

#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convit.py
ConVit_B_model = lambda device:  Model(model=timm.create_model('convit_base', pretrained=True), crop_pct=1.0, device=device)
ConVit_S_model = lambda device:  Model(model=timm.create_model('convit_small', pretrained=True), crop_pct=1.0, device=device)

Clip_vit_B16_model =  lambda device:  ClipEnsemble( 'ViT-B/16', imagenet_classes_for_clip, imagenet_single_template, device) 
Clip_RN50_model = lambda device: ClipEnsemble( "RN50", imagenet_classes_for_clip, imagenet_single_template, device) 
Clip_RN101_model = lambda device: ClipEnsemble( "RN101", imagenet_classes_for_clip, imagenet_single_template, device) 

Clip_vit_B16_ensemble_80_model = lambda device:  ClipEnsemble( 'ViT-B/16', imagenet_classes_for_clip, imagenet_templates, device) 
Clip_RN50_ensemble_80_model = lambda device:  ClipEnsemble( "RN50", imagenet_classes_for_clip, imagenet_templates, device) 
Clip_RN101_ensemble_80_model = lambda device:  ClipEnsemble( "RN101", imagenet_classes_for_clip, imagenet_templates, device) 

Clip_vit_B16_ensemble_pose_model = lambda device:  ClipEnsemble( 'ViT-B/16', imagenet_classes_for_clip, imagenet_pose_templates, device) 
Clip_RN50_ensemble_pose_model = lambda device:  ClipEnsemble( "RN50", imagenet_classes_for_clip, imagenet_pose_templates, device) 
Clip_RN101_ensemble_pose_model = lambda device:  ClipEnsemble( "RN101", imagenet_classes_for_clip, imagenet_pose_templates, device) 


ViT_S16_model = lambda device:  Model(model=timm.create_model('vit_small_patch16_224', pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)
ViT_B16_sam_model = lambda device:  Model(model=timm.create_model('vit_base_patch16_224_sam', pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)
ViT_B16_model = lambda device:  Model(model=timm.create_model('vit_base_patch16_224', pretrained=True), crop_pct = 0.9, interpolation=Image.BICUBIC, device=device)
ViT_L16_model = lambda device:  Model(model=timm.create_model("vit_large_patch16_224", pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
SWIN_B_model = lambda device:  Model(model=timm.create_model('swin_base_patch4_window7_224', pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)
SWIN_B_384_model = lambda device:  Model(model=timm.create_model('swin_base_patch4_window12_384', pretrained=True), input_size=384, crop_pct=1.0, interpolation=Image.BICUBIC, device=device)
SWIN_L_model = lambda device:  Model(model=timm.create_model('swin_large_patch4_window7_224', pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)
SWIN_L_384_model = lambda device:  Model(model=timm.create_model('swin_large_patch4_window12_384', pretrained=True), input_size=384, crop_pct=1.0, interpolation=Image.BICUBIC, device=device)

#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnetv2.py
BiTM_RN50_448_model = lambda device:  Model(model=timm.create_model('resnetv2_50x1_bitm', pretrained=True), input_size=448, crop_pct=1.0, device=device)
BiTM_RN101_448_model = lambda device:  Model(model=timm.create_model('resnetv2_101x1_bitm', pretrained=True), input_size=448, crop_pct=1.0, device=device)
BiTM_RN152_448_model = lambda device:  Model(model=timm.create_model('resnetv2_152x2_bitm', pretrained=True), input_size=448, crop_pct=1.0, device=device)

Simclr_model = lambda device:  Model(model=simclr_resnet50x1(pretrained=True, use_data_parallel=True), device=device)

#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
SWSL_ResNeXt101_model = lambda device:  Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl'), device=device) #ResNeXt101_32x16d_swsl('ResNeXt101_32x16d_swsl')
SWSL_RN50_model = lambda device:  Model(model=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl'), device=device)#resnet50_swsl('resnet50_swsl')


#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/deit.py
Deit_B16_224_model =lambda device:  Model(model= timm.create_model('deit_base_distilled_patch16_224', pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)
Deit_S16_224_model = lambda device:  Model(model=timm.create_model('deit_small_distilled_patch16_224', pretrained=True), crop_pct=0.9, interpolation=Image.BICUBIC, device=device)

#https://github.com/facebookresearch/SWAG/blob/main/imagenet_1k_eval.py
SWAG_ViT_L16_512_model = lambda device: Model(torch.hub.load("facebookresearch/swag", model="vit_l16_in1k"), input_size=512, crop_pct=1.0, interpolation=transforms.InterpolationMode.BICUBIC, device=device)
SWAG_ViT_H14_518_model = lambda device: Model(torch.hub.load("facebookresearch/swag", model="vit_h14_in1k"), input_size=518, crop_pct=1.0, interpolation=transforms.InterpolationMode.BICUBIC, device=device)
SWAG_RegNetY_128GF_384_model = lambda device: Model(torch.hub.load("facebookresearch/swag", model="regnety_128gf_in1k"), input_size=384, crop_pct=1.0, interpolation=transforms.InterpolationMode.BICUBIC, device=device)

EffN_l2_ns_475_model = lambda device: Model(torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_l2_ns_475", pretrained=True),
                                              interpolation=transforms.InterpolationMode.BICUBIC, input_size=475, crop_pct = 0.936, device=device,)
EffN_l2_ns_800_model = lambda device: Model(torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_l2_ns", pretrained=True),
                                              interpolation=transforms.InterpolationMode.BICUBIC, input_size=800, crop_pct = 0.96, device=device,)
EffN_b7_ns_600_model = lambda device: Model(model=torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_b7_ns", pretrained=True),
                                              interpolation=transforms.InterpolationMode.BICUBIC, input_size=600, crop_pct =  0.949, device=device,)
                                              
ViT_21k_B16_model = lambda device: Model(ViT('B_16_imagenet1k', pretrained=True), input_size=384, crop_pct=1.0,
                                               device=device, in_mean=[0.5, 0.5, 0.5], in_std=[0.5, 0.5, 0.5])

ViT_21k_L16_model = lambda device: Model(ViT('L_16_imagenet1k', pretrained=True), input_size=384, crop_pct=1.0,
                                              device=device, in_mean=[0.5, 0.5, 0.5], in_std=[0.5, 0.5, 0.5])
#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py
Mixer_B16_224_model = lambda device: Model(timm.create_model('mixer_b16_224', pretrained=True),
                                              interpolation=transforms.InterpolationMode.BICUBIC, device=device, in_mean=[0.5, 0.5, 0.5], in_std=[0.5, 0.5, 0.5])
Mixer_L16_224_model = lambda device: Model(timm.create_model('mixer_l16_224', pretrained=True),
                                              interpolation=transforms.InterpolationMode.BICUBIC, device=device, in_mean=[0.5, 0.5, 0.5], in_std=[0.5, 0.5, 0.5])
class BEiT(Model):
    def __init__(self, model, device, transform=None, in_mean=[0.485, 0.456, 0.406], in_std=[0.229, 0.224, 0.225], input_size=224, crop_pct=1.0, interpolation=Image.BILINEAR):
        super().__init__(model, device, transform, in_mean, in_std, input_size, crop_pct, interpolation)

    def __call__(self, batch):
        '''
        
        '''
        with torch.no_grad():
            self.model.eval()
            output = self.model(batch).logits
            # probs = output.softmax(dim=-1)

        return output

def Beit_B16_224_model(device):
    '''
        BeitFeatureExtractor {
        "crop_size": 224,
        "do_center_crop": false,
        "do_normalize": true,
        "do_resize": true,
        "feature_extractor_type": "BeitFeatureExtractor",
        "image_mean": [0.5,0.5,0.5],
        "image_std": [0.5,0.5,0.5],
        "reduce_labels": false,
        "resample": 2,
        "size": 224
        }
    '''
  
    model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")         
    beit_base_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
    beit_transform= transforms.Compose( [ 
          lambda images: beit_base_feature_extractor(images, return_tensors="pt")['pixel_values'].squeeze(0) ])
    return BEiT(model=model, transform=beit_transform, device=device)

def Beit_L16_224_model(device):
    '''
         BeitFeatureExtractor {
        "crop_size": 224,
        "do_center_crop": false,
        "do_normalize": true,
        "do_resize": true,
        "feature_extractor_type": "BeitFeatureExtractor",
        "image_mean":[0.5,0.5,0.5],
        "image_std":[0.5,0.5,0.5],
        "reduce_labels": false,
        "resample": 2,
        "size": 224
        }
    '''

    model = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224") 
    beit_large_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
    beit_transform= transforms.Compose( [ 
          lambda images: beit_large_feature_extractor(images, return_tensors="pt")['pixel_values'].squeeze(0) ])
    return BEiT(model=model, transform=beit_transform, device=device)




models_names_dict = {
            #    'NS_EffN_l2_800' : [EffN_l2_ns_800_model, 'Noisy_Student_EfficientNet_L2_800_noisy_student', 'torchhub',  800],
               'ViT_21k_L16': [ViT_21k_L16_model, 'ViT_21k_Large16_384', 'pytorch_pretrained_vit',  384], 'ViT_21k_B16': [ViT_21k_B16_model, 'ViT_21k_Base16_384', 'pytorch_pretrained_vit', 384], 
               'SWAG_ViT_L16_512':[SWAG_ViT_L16_512_model, 'SWAG_Large16_512', 'torchhub', 512], 'SWAG_ViT_H14_512':[SWAG_ViT_H14_518_model, 'SWAG_Huge14_512', 'torchhub', 518],
               'SWAG_RegNetY_128GF_384':[SWAG_RegNetY_128GF_384_model, 'SWAG_RegNetY_128GF_384', 'torchhub', 512],

               'NS_EffN_l2_475' : [EffN_l2_ns_475_model, 'Noisy_Student_EfficientNet_L2_475', 'torchhub',  475], 
               'NS_EffN_b7_600' : [EffN_b7_ns_600_model, 'Noisy_Student_EfficientNet_B7_600', 'torchhub',  600], 

               'ResNet50':[Resnet50_model, 'ResNet50', 'torchvision', 224], 'ResNet101':[Resnet101_model, 'ResNet101', 'torchvision', 224], 'ResNet152':[Resnet152_model, 'ResNet152', 'torchvision', 224],

               'SWIN_L_384': [SWIN_L_384_model, 'SWIN_Large_patch4_window12_384', 'timm', 384], 'SWIN_B_384': [SWIN_B_384_model, 'SWIN_Base_patch4_window12_384', 'timm', 384], 
               'SWIN_B': [SWIN_B_model, 'SWIN_Base_patch4_window7_224', 'timm',  224], 'SWIN_L': [SWIN_L_model, 'SWIN_Large_patch4_window7_224', 'timm',  224],
               'ViT_S':[ViT_S16_model, 'ViT_Small_patch16_224', 'timm', 224], 'ViT_B':[ViT_B16_model, 'ViT_Base_patch16_224', 'timm',  224], 'ViT_L':[ViT_L16_model, 'ViT_large_patch16_224', 'timm',  224], 'ViT_B16_sam':[ViT_B16_sam_model, 'ViT_Base_patch16_sam_224', 'timm',  224],
               'BiTM_RN50_448':[BiTM_RN50_448_model, 'BiTM_ResNetv2_50x1_448', 'timm', 448], 'BiTM_RN101_448':[BiTM_RN101_448_model, 'BiTM_ResNetv2_101x1_448_448', 'timm',  448], 'BiTM_RN152x2_448':[BiTM_RN152_448_model, 'BiTM_ResNetv2_152x2_448', 'timm',  224],
               'SWSL_ResNet50':[SWSL_RN50_model, 'SWSL_ResNet50', 'torchhub', 224], 'SWSL_ResNeXt101':[SWSL_ResNeXt101_model, 'SWSL_ResNeXt101', 'torchhub',  224], 
               'Mixer_B16':[Mixer_B16_224_model, 'Mixer_Base_16_224', 'timm',  224], 'Mixer_L16':[Mixer_L16_224_model, 'Mixer_large_16_224', 'timm',  224],
               'BEiT_B16' : [Beit_B16_224_model, 'BEiT_B16_224', 'timm',  224], 'BEiT_L16' : [Beit_L16_224_model, 'BEiT_L16_224', 'timm',  224],
               'Deit_B16' : [Deit_B16_224_model, 'Deit_B16_224', 'timm',  224], 'Deit_S16' : [Deit_S16_224_model, 'Deit_S16_224', 'timm',  224],

               'SimCLR': [Simclr_model, 'SimCLR_ResNet50', 'ptrnets',  224], 
               'ConvNext_XL_model': [ConvNext_XL_model, 'ConvNext_xlarge_in22ft1k', 'timm',  224],
               'ConvNext_L_in_model': [ConvNext_L_in_model, 'ConvNext_large', 'timm',  224],
               'ConvNext_B_384_model': [ConvNext_B_384_model, 'ConvNeXt_Base_384_in22ft1k', 'timm', 384], 'ConvNext_B_model': [ConvNext_B_model, 'ConvNeXt_bBse_in22ft1k', 'timm',  224],
               'ConvNeXt_L_384_model': [ConvNext_L_384_model, 'ConvNeXt_Large_384_in22ft1k', 'timm', 384], 'ConvNeXt_L_model': [ConvNext_L_model, 'ConvNeXt_Large_in22ft1k', 'timm', 224], 
               
               'ConVit_B_model': [ConVit_B_model, 'ConViT_Base', 'timm',  224], 'ConVit_S_model': [ConVit_S_model, 'ConVit_Small', 'timm',  224],
    

               'CLIP_ViT_B16':[Clip_vit_B16_model, 'CLIP_ViT_Base_16', 'CLIP',  224], 'CLIP_50':[Clip_RN50_model, 'CLIP_ResNet50', 'CLIP',  224], 'CLIP_101':[Clip_RN101_model, 'CLIP_ResNet101', 'CLIP',  224],
               'CLIP_ViT_B16_ens_80':[Clip_vit_B16_ensemble_80_model, 'CLIP_ViT_Base_16_ensemble_80', 'CLIP_ensemble',  224], 'CLIP_50_ensemble_80':[Clip_RN50_ensemble_80_model, 'CLIP_ResNet50_ensemble_80', 'CLIP_ensemble',  224], 'CLIP_101_ensemble_80':[Clip_RN101_ensemble_80_model, 'CLIP_ResNet101_ensemble_80', 'CLIP_ensemble',  224],
               'CLIP_ViT_B16_pose_ens':[Clip_vit_B16_ensemble_pose_model, 'CLIP_ViT_Base_16_pose_ensemble', 'CLIP_ensemble',  224], 'CLIP_50_pose_ens':[Clip_RN50_ensemble_pose_model, 'CLIP_ResNet50_ensemble', 'CLIP_ensemble',  224], 'CLIP_101_pose_ens':[Clip_RN101_ensemble_pose_model, 'CLIP_ResNet101_pose_ensemble', 'CLIP_ensemble',  224],

}

