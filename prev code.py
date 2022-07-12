
# Resnet50_model = lambda device: torchvision.models.resnet50(pretrained=True)
# Resnet152_model = lambda device: torchvision.models.resnet152(pretrained=True)
# Resnet101_model = lambda device: torchvision.models.resnet101(pretrained=True)

# ConvNext_XL_model = lambda device: timm.create_model('ConvNext_xlarge_in22ft1k', pretrained=True)
# ConvNext_L_384_model = lambda device: timm.create_model('ConvNext_large_384_in22ft1k', pretrained=True)
# ConvNext_L_model = lambda device: timm.create_model('ConvNext_large_in22ft1k', pretrained=True)
# ConvNext_B_384_model= lambda device: timm.create_model('ConvNext_base_384_in22ft1k', pretrained=True)
# ConvNext_B_model = lambda device: timm.create_model('ConvNext_base_in22ft1k', pretrained=True)
# ConvNext_L_in_model = lambda device: timm.create_model('ConvNext_large', pretrained=True)


# ConVit_B_model = lambda device: timm.create_model('convit_base', pretrained=True)
# ConVit_S_model = lambda device: timm.create_model('convit_small', pretrained=True)

# Clip_vit_B16_model = lambda device: clip.load('ViT-B/16', device=device)[0]
# Clip_RN50_model = lambda device:clip.load("RN50", device=device)[0]
# Clip_RN101_model = lambda device:clip.load("RN101", device=device)[0]

# Clip_vit_B16_ensemble_80_model = lambda device:  ClipEnsemble( 'ViT-B/16', imagenet_classes_for_clip, imagenet_templates, device) 
# Clip_RN50_ensemble_80_model = lambda device:  ClipEnsemble( "RN50", imagenet_classes_for_clip, imagenet_templates, device) 
# Clip_RN101_ensemble_80_model = lambda device:  ClipEnsemble( "RN101", imagenet_classes_for_clip, imagenet_templates, device) 

# Clip_vit_B16_ensemble_model = lambda device:  ClipEnsemble( 'ViT-B/16', imagenet_classes_for_clip, imagenet_pose_templates, device) 
# Clip_RN50_ensemble_model = lambda device:  ClipEnsemble( "RN50", imagenet_classes_for_clip, imagenet_pose_templates, device) 
# Clip_RN101_ensemble_model = lambda device:  ClipEnsemble( "RN101", imagenet_classes_for_clip, imagenet_pose_templates, device) 

# ViT_S16_model = lambda device: timm.create_model('vit_small_patch16_224', pretrained=True)
# ViT_B16_sam_model = lambda device: timm.create_model('vit_base_patch16_224_sam', pretrained=True)
# ViT_B16_model = lambda device: timm.create_model('vit_base_patch16_224', pretrained=True)
# ViT_L16_model = lambda device: timm.create_model("vit_large_patch16_224", pretrained=True)

# ViT_21k_B16_model = lambda device: ViT('B_16_imagenet1k', pretrained=True)
# ViT_21k_L16_model = lambda device: ViT('L_16_imagenet1k', pretrained=True)

# SWIN_B_model = lambda device: timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# SWIN_B_384_model = lambda device: timm.create_model('swin_base_patch4_window12_384', pretrained=True)
# SWIN_L_model = lambda device: timm.create_model('swin_large_patch4_window7_224', pretrained=True)
# SWIN_L_384_model = lambda device: timm.create_model('swin_large_patch4_window12_384', pretrained=True)

# BiTM_RN50_model = lambda device: timm.create_model('resnetv2_50x1_bitm', pretrained=True)
# BiTM_RN101_model = lambda device: timm.create_model('resnetv2_101x1_bitm', pretrained=True)
# BiTM_RN152_model = lambda device: timm.create_model('resnetv2_152x2_bitm', pretrained=True)
# EffN_b7_ns_model = lambda device: timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
# EffN_l2_ns_model = lambda device: timm.create_model('tf_efficientnet_l2_ns', pretrained=True)

# EffN_l2_ns_475_model = lambda device: torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_l2_ns_475", pretrained=True)
# EffN_l2_ns_800_model = lambda device: torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_l2_ns", pretrained=True)

# EffN_b7_ns_600_model = lambda device: torch.hub.load("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_b7_ns", pretrained=True)


# Simclr_model = lambda device: simclr_resnet50x1(pretrained=True, use_data_parallel=True)

# SWSL_ResNeXt101_model = lambda device: torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')  #ResNeXt101_32x16d_swsl('ResNeXt101_32x16d_swsl')
# SWSL_RN50_model = lambda device: torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl') #resnet50_swsl('resnet50_swsl')

# Mixer_B16_224_model = lambda device: timm.create_model('mixer_b16_224', pretrained=True)
# Mixer_L16_224_model = lambda device: timm.create_model('mixer_l16_224', pretrained=True)
# Mixer_S16_224_model = lambda device: timm.create_model('mixer_s16_224', pretrained=True)


# Beit_B16_224_model = lambda device: BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224") #timm.create_model('beit_base_patch16_224', pretrained=True) #beit_base_patch16_224_pt22k_ft22kto1k
# Beit_L16_224_model = lambda device: BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224") #timm.create_model('beit_large_patch16_224', pretrained=True)

# Deit_B16_224_model =lambda device:  timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
# Deit_S16_224_model = lambda device: timm.create_model('deit_small_distilled_patch16_224', pretrained=True)

# models_names_dict = {
#                'Resnet50':[Resnet50_model, 'Resnet50', 'torchvision', 32, 224], 'Resnet101':[Resnet101_model, 'Resnet101', 'torchvision', 32, 224], 'Resnet152':[Resnet152_model, 'Resnet152', 'torchvision', 32, 224],
#                'ConvNext_L_384_model': [ConvNext_L_384_model, 'ConvNext_large_384_in22ft1k', 'timm', 4, 384], 'ConvNext_L_model': [ConvNext_L_model, 'ConvNext_large_in22ft1k', 'timm', 32, 224], 
#                'SWIN_L_384': [SWIN_L_384_model, 'swin_large_patch4_window12_384', 'timm', 8, 384], 'SWIN_B_384': [SWIN_B_384_model, 'swin_base_patch4_window12_384', 'timm', 8, 384], 
#                'ViT_21k_B16': [ViT_21k_B16_model, 'ViT_21k_base16_384', 'pytorch_pretrained_vit', 8, 384], 'ViT_21k_L16': [ViT_21k_L16_model, 'ViT_21k_large16_384', 'pytorch_pretrained_vit', 8, 384],
#                'ConvNext_B_384_model': [ConvNext_B_384_model, 'ConvNext_base_384_in22ft1k', 'timm', 8, 384], 'ConvNext_B_model': [ConvNext_B_model, 'ConvNext_base_in22ft1k', 'timm', 32, 224],
#                'vit_S':[ViT_S16_model, 'vit_small_patch16_224', 'timm', 32, 224], 'vit_B':[ViT_B16_model, 'vit_base_patch16_224', 'timm', 32, 224], 'vit_L':[ViT_L16_model, 'vit_large_patch16_224', 'timm', 32, 224], 'vit_B16_sam':[ViT_B16_sam_model, 'vit_base_patch16_sam_224', 'timm', 32, 224],
#                'Resnet50':[Resnet50_model, 'Resnet50', 'torchvision', 32, 224], 'Resnet101':[Resnet101_model, 'Resnet101', 'torchvision', 32, 224], 'Resnet152':[Resnet152_model, 'Resnet152', 'torchvision', 32, 224],
#                'BiTM_RN50':[BiTM_RN50_model, 'BiTM_resnetv2_50x1', 'timm', 32, 224], 'BiTM_RN101':[BiTM_RN101_model, 'BiTM_resnetv2_101x1', 'timm', 32, 224], 'BiTM_RN152x2':[BiTM_RN152_model, 'BiTM_resnetv2_152x2', 'timm', 32, 224],
#                'SWSL_ResNet50':[SWSL_RN50_model, 'SWSL_ResNet50', 'torchhub', 32, 224], 'SWSL_ResNeXt101':[SWSL_ResNeXt101_model, 'SWSL_ResNeXt101', 'torchhub', 32, 224], 
#                'Mixer_B16':[Mixer_B16_224_model, 'Mixer_base_16_224', 'timm', 32, 224], 'Mixer_L16':[Mixer_L16_224_model, 'Mixer_large_16_224', 'timm', 32, 224],
#                'Beit_B16' : [Beit_B16_224_model, 'Beit_B16_224', 'timm', 32, 224], 'Beit_L16' : [Beit_L16_224_model, 'Beit_L16_224', 'timm', 32, 224],
#                'Deit_B16' : [Deit_B16_224_model, 'Deit_B16_224', 'timm', 32, 224], 'Deit_S16' : [Deit_S16_224_model, 'Deit_S16_224', 'timm', 32, 224],
#                'Simclr': [Simclr_model, 'SimCLR_ResNet50', 'ptrnets', 32, 224], 
#                'SWIN_B': [SWIN_B_model, 'swin_base_patch4_window7_224', 'timm', 32, 224], 'SWIN_L': [SWIN_L_model, 'swin_large_patch4_window7_224', 'timm', 32, 224],
#                'ConvNext_XL_model': [ConvNext_XL_model, 'ConvNext_xlarge_in22ft1k', 'timm', 32, 224],
#                'ConvNext_L_in_model': [ConvNext_L_in_model, 'ConvNext_large', 'timm', 32, 224],
#                'ConVit_B_model': [ConVit_B_model, 'convit_base', 'timm', 32, 224], 'ConVit_S_model': [ConVit_S_model, 'convit_small', 'timm', 32, 224],
    
#                'ViT_21k_B16': [ViT_21k_B16_model, 'ViT_21k_base16_384', 'pytorch_pretrained_vit', 8, 384], 'ViT_21k_L16': [ViT_21k_L16_model, 'ViT_21k_large16_384', 'pytorch_pretrained_vit', 8, 384],
#                'Mixer_B16':[Mixer_B16_224_model, 'Mixer_base_16_224', 'timm', 64, 224], 'Mixer_L16':[Mixer_L16_224_model, 'Mixer_large_16_224', 'timm', 64, 224],
#                'EffN_l2_ns_475' : [EffN_l2_ns_475_model, 'Efficientnet_l2_475_noisy_student', 'torchhub', 16, 475], 

#                'EffN_b7_ns_600' : [EffN_b7_ns_600_model, 'Efficientnet_b7_600_noisy_student', 'torchhub', 16, 600], 
#                'EffN_l2_ns_800' : [EffN_l2_ns_800_model, 'Efficientnet_l2_800_noisy_student', 'torchhub', 8, 800],

#                'Clip_vit_B16':[Clip_vit_B16_model, 'CLIP_ViT_base_16', 'clip', 32, 224], 'Clip_50':[Clip_RN50_model, 'CLIP_ResNet50', 'clip', 32, 224], 'CLIP_101':[Clip_RN101_model, 'CLIP_ResNet101', 'clip', 32, 224],
#                'Clip_vit_B16_ensemble_80':[Clip_vit_B16_ensemble_80_model, 'CLIP_ViT_base_16_ensemble_80', 'clip_ensemble', 64, 224], 'Clip_50_ensemble_80':[Clip_RN50_ensemble_80_model, 'CLIP_ResNet50_ensemble_80', 'clip_ensemble', 64, 224], 'CLIP_101_ensemble_80':[Clip_RN101_ensemble_80_model, 'CLIP_ResNet101_ensemble_80', 'clip_ensemble', 64, 224],
#                'Clip_vit_B16_ensemble':[Clip_vit_B16_ensemble_pose_model, 'CLIP_ViT_base_16_ensemble', 'clip_ensemble', 64, 224], 'Clip_50_ensemble':[Clip_RN50_ensemble_pose_model, 'CLIP_ResNet50_ensemble', 'clip_ensemble', 64, 224], 'CLIP_101_ensemble':[Clip_RN101_ensemble_pose_model, 'CLIP_ResNet101_ensemble', 'clip_ensemble', 64, 224],


# }




#############




# from datetime import datetime
# from models.CLIP import clip
# from torch.utils.data import  DataLoader
# from torchvision import transforms
# import matplotlib.pyplot as plt; plt.rcdefaults()
# from models.models import models_names_dict
# from run_model import run_model
# from imagenet_classes import imagenet_classes
# import torch
# import argparse
# from utils import seed_everything, report_writer, get_logger
# from dataloader import ImageNetV2Dataset, get_inv2_data_tranforms

# # cd dl/AMMIGP/ObjectPose/src

# ##satrting form src/

# # cd ../..
# # rm -rf ObjectPose
# # git clone https://github.com/amro-kamal/ObjectPose.git
# # cd ObjectPose/src/models
# # git clone https://github.com/openai/CLIP
# # cd ..

# # python run_imagenet_v2.py

# # git add .
# # git commit -m 'added torch version to req.txt'
# # git push


# def run_imagenet_v2(models, batch_sizes, input_sizes, models_names, model_zoos, datapath='gdrive/MyDrive/AMMI GP/newdata/imagenetv2-matched-frequency', all_result_path=''):
#     print('Code update 7')
#     writer = report_writer(all_result_path, 'imagenet_v2')
#     writer.write(f'\n Experiment: new data transforms {datetime.now().time().strftime("%H:%M:%S")}, {datetime.now().strftime("%Y-%m-%d")} \n')
#     for  m in range(len(models)):
#         print(f'model {m+1}/{len(models)}: {models_names[m]}')
#         models_accs = {}
#         batch_size = batch_sizes[m]
#         input_size = input_sizes[m]
#         model_name = model_names[m]
#         model_zoo=model_zoos[m]
#         # print('getting the model')
#         model = models[m]()
#         print('model is ready')
#         correct1 = 0
#         correct5 = 0
#         total = 0
        
#         data_transform = get_inv2_data_tranforms(model_zoo, model_name, input_size )

#         # print('dataset')
#         imagenetv2 = ImageNetV2Dataset(root=datapath, transform=data_transform) 
#         # print('dataloader')
#         test_loader = DataLoader(imagenetv2, batch_size=batch_size, shuffle=False, num_workers=1)
#         # print('starting the loop')
#         with torch.no_grad():
#             for batch, target in test_loader:
#                 target = target.to(device)
#                 # print('target shape ', target.shape)
#                 if model_zoo in ['torchvision', 'timm', 'ptrnets', 'torchhub', 'pytorch_pretrained_vit']: #torchvision model
#                     # print('feed the batch')
#                     model.eval()
#                     model = model.to(device) 
#                 output = model(batch.to(device))
#                     # probs = output.softmax(dim=-1)

#                 elif model_zoo=="modelvshuman": #modelvshuman
#                     output = model.forward_batch(batch.to(device))
#                     output = torch.tensor(output)
#                     # probs = output.softmax(dim=-1)

#                 elif model_zoo =='clip_ensemble':
#                     image_input = batch.to(device)
#                     similarity = model.forward_batch(image_input)
#                     output = similarity
#                     # probs = similarity.softmax(dim=-1) #normalized similarity

#                 elif model_zoo=='clip':
#                     image_input = batch.to(device) 
#                     text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in list(imagenet_classes.values())]).to(device) 
#                     with torch.no_grad():
#                       image_features = model.encode_image(image_input)
#                       text_features = model.encode_text(text_inputs)
#                     # Pick the top 5 most similar labels for the image
#                     image_features /= image_features.norm(dim=-1, keepdim=True)
#                     text_features /= text_features.norm(dim=-1, keepdim=True)
#                     similarity = (100.0 * image_features @ text_features.T)
#                     output = similarity
#                     # probs = similarity.softmax(dim=-1) #normalized similarity
                
#                 #top5
#                 # print('calculate acc')
#                 _, predicted = output.topk(5, 1, True, True)
#                 predicted = predicted.t()
#                 # print('predited , target', predicted.shape, target.shape)
                
#                 correct = predicted.eq(target.view(1, -1).expand_as(predicted))
#                 # print('correct ',correct.shape)
#                 correct1 += correct[:1].reshape(-1).float().sum(0)
#                 correct5 += correct[:5].reshape(-1).float().sum(0)
#                 total += target.size(0)

#         # print('del model')
#         del model
#         accuracy1 = (correct1 / total)*100
#         accuracy5 = (correct5 / total)*100

#         print(f'{model_name}: top1: {accuracy1}, top5: {accuracy5}')
#         # logger.info(f'{model_name}: top1 accuracy: {accuracy1}, top5 accuracy: {accuracy5}')
#         writer.write(f'{model_name}: top1: {accuracy1}, top5: {accuracy5} \n')

#         models_accs[model_name] = {'top1':accuracy1, 'top5':accuracy5}
        
#         # pickle_file = open(os.path.join(all_result_path, model_name+'_all_results.pkl'), "wb")
#         # pickle.dump(models_accs, pickle_file) 





# parser = argparse.ArgumentParser(description='imagenetv2-matched-frequency')

# parser.add_argument('--dataroot', default = '../../data/imagenetv2-matched-frequency/data', type=str)
# parser.add_argument('--allresultpath', default = '../../data/imagenetv2-matched-frequency/all_results', type=str)

# parser.add_argument('--seed', default=42, type=int)

# args=parser.parse_args()
# seed_everything(seed=args.seed)

# if __name__=='__main__':

#     # telecolab.start('Calibration: resnet50')
    
#     models = [ model[0] for model in list(models_names_dict.values())]
#     model_names = [ model[1] for model in list(models_names_dict.values())]
#     model_zoos = [ model[2] for model in list(models_names_dict.values())]
#     batch_sizes = [ model[3] for model in list(models_names_dict.values())]
#     input_sizes = [ model[4] for model in list(models_names_dict.values())]

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f'Running on {device}')

#     datapath = args.dataroot #'../../CO3D_subset'
#     all_result_path = args.allresultpath



#     run_imagenet_v2(models, batch_sizes, input_sizes, model_names, model_zoos, datapath=datapath, all_result_path=all_result_path)


#######################


# from torch.utils.data import  DataLoader
# import os
# from torchvision import transforms
# import numpy as np
# import matplotlib.pyplot as plt; plt.rcdefaults()
# import pickle
# from models.models import models_names_dict
# from dataloader import ImagePoseData
# from run_model import run_model
# from imagenet_classes import imagenet_classes
# import torch
# import argparse
# from .utils import seed_everything


# AIRLINER=404;  BARBERCHAIR = 423; CANNON=471; FIREENGINE=555; FOLDINGCHAIR=559; FORKLIFT=561;
# GARBAGETRUCK = 569; HAMMERHEAD=4; JEEP=609; MOUNTAINBIKE = 671; PARKBENCH=703; ROCKINGCHAIR=765; SHOPPINGCART=791;
# TABLELAMP=846; LAMPSHADE=619; TANK = 847;  TRACTOR = 866; WHEELBARROW = 428; MOPED= 665; MOTORSCOOTER = 670;
# TANDEM_BICYCLE = 444; PARKINGMETER=704; TOILET=861; TOASTER=859; BROCCOLI= 937; BANANA=954; MICROWAVE=651; HOTDOG=934

# parser = argparse.ArgumentParser(description='CO3D')

# parser.add_argument('--dataroot', default = '../../data/CO3D_subset', type=str)
# parser.add_argument('--seed', default=42, type=int)

# args=parser.parse_args()
# seed_everything(seed=args.seed)

# if __name__=='__main__':

#     # telecolab.start('Calibration: resnet50')
    
#     models = [ model[0] for model in list(models_names_dict.values())]
#     model_names = [ model[1] for model in list(models_names_dict.values())]
#     model_zoos = [ model[2] for model in list(models_names_dict.values())]
#     batch_sizes = [ model[3] for model in list(models_names_dict.values())]
#     input_sizes = [ model[4] for model in list(models_names_dict.values())]

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dataroot = args.dataroot #'../../CO3D_subset'




#     #################################
#     #################################

#     true_class_dict = {'banana' : [BANANA]} #, 'microwave': [MICROWAVE], 'hotdog':[HOTDOG]} #{'toilet': [TOILET], 'toaster':[TOASTER], 'parkingmeter': [PARKINGMETER], 'bench': [PARKBENCH], 'bicycle':[MOUNTAINBIKE], 'motorscooter': [MOTORSCOOTER] } #{'bench': [PARKBENCH], 'motorcycle': [MOPED, MOTORSCOOTER], 'bicycle':[MOUNTAINBIKE] }
#     pose = [None]
#     objects = list(true_class_dict.keys())

#     #################################
#     #################################
#     accs_and_confs = {}
  
#     for  m in range(len(models)):
#         batch_size = batch_sizes[m]
#         input_size = input_sizes[m]
#         # telecolab.send(messages=f'Starting on model number {m+1}/{len(models)}: {models[m]}')

#         all_result_path = f'{dataroot}/all_result/{model_names[m]}' #CO3D_rot_both360 #path to save the allreslt.pkl file #f'gdrive/MyDrive/AMMI GP/data/CO3D dataset/all_result/{model_names[m]}' #f'{dataroot}/all_result/{model_names[m]}' 
#         if not os.path.exists(f'{dataroot}/all_result'):
#             os.mkdir(f'{dataroot}/all_result')
#         if not os.path.exists(f'{dataroot}/all_result/{model_names[m]}'):
#             os.mkdir(f'{dataroot}/all_result/{model_names[m]}')
#         accs_and_confs[model_names[m]] = {}

#         # rotboth ->vil_L->8, SWIN_384 -> 16  / the rest 32
#         # report = report_writer(os.path.join(data_root_path[0], f'model_result'), obj_name=obj)
#         for obj in objects:
#             accs_and_confs[model_names[m]][obj] = {}
#             print('current model : ',model_names[m])
#             print('✅✅✅✅✅✅✅✅✅✅✅✅✅ object: ', obj)
#             true_class = true_class_dict[obj]

#             data_root_path = [f'{dataroot}/data/{obj}'] #bench data' #'gdrive/MyDrive/AMMI GP/data/CO3D dataset/data'
#             ############################################################################################################################
#             print('current model : ',model_names[m])
#             savepath = [f'{dataroot}/data/{obj}/model_result/{model_names[m]}'] #f'gdrive/MyDrive/AMMI GP/data/CO3D dataset/model_result/{model_names[m]}'
           
# ############################################################################################################################

#             all_labels, all_logits, bg_num, all_correct = [], [], 0, 0
#             for i in range(len(data_root_path)):

#                 if not os.path.exists(f'{dataroot}/data/{obj}/model_result'):  #rot_both360
#                   os.mkdir(f'{dataroot}/data/{obj}/model_result')
#                 if not os.path.exists(f'{dataroot}/data/{obj}/model_result/{model_names[m]}'):
#                   os.mkdir(f'{dataroot}/data/{obj}/model_result/{model_names[m]}')


#                 data_transform = transforms.Compose( [ 
#                           # transforms.Scale(224),
#                           transforms.Resize(size=input_size),
#                           transforms.CenterCrop(size=(input_size, input_size)),
#                           transforms.ToTensor(),
#                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                           ])
                
#                 data = ImagePoseData(os.path.join(data_root_path[i], f'images'), transform=data_transform) #rot_both_images
#                 mydataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
#               ######################################################################
#               # run the model and save the log.txt and result.txt files to the savepath
#                 correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, result, result5, logits, labels =  run_model(models[m](), mydataloader,
#                                                                                         list(imagenet_classes.values()),savepath[i], 
#                                                                                     pose=pose[i], model_name=model_names[m], model_zoo=model_zoos[m],
#                                                                                     true_classes=true_class, report=None, device=device)
#               #####################################################################
                                    
#             #Calibration
#             accs_and_confs[model_names[m]][obj]= {'logits':logits.tolist(), 'labels':labels.tolist(), 'correct':correct}
#             pickle_file = open(os.path.join(all_result_path, model_names[m]+'_all_results.pkl'), "wb")
#             pickle.dump(accs_and_confs, pickle_file) 
#         # log = open (os.path.join(os.path.join( all_result_path, model_names[m]+'_all_results.yml' ), "w") #summary table



########################

import torch
from torch.utils.data import  DataLoader
import os
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import pickle
from .models.models import models_names_dict
from dataloader import ImagePoseData
from run_model import run_model
import argparse
from imagenet_classes import imagenet_classes
from .utils import seed_everything

AIRLINER=404;  BARBERCHAIR = 423; CANNON=471; FIREENGINE=555; FOLDINGCHAIR=559; FORKLIFT=561;
GARBAGETRUCK = 569; HAMMERHEAD=4; JEEP=609; MOUNTAINBIKE = 671; PARKBENCH=703; ROCKINGCHAIR=765; SHOPPINGCART=791;
TABLELAMP=846; LAMPSHADE=619; TANK = 847;  TRACTOR = 866; WHEELBARROW = 428; 



parser = argparse.ArgumentParser(description='CO3D')

parser.add_argument('--dataroot', default = '../../data/360/rot_rand_roll', type=str, help='Path to ObjectPose data.')
parser.add_argument('--bgs', default = 'bg1 bg2 nobg', type=str, help='The backgrounds.')
parser.add_parser('poses', default='ROLL YAW PITCH', type=str, help='The rotation axes.')
parser.add_argument('--seed', default=42, type=int)

args=parser.parse_args()
seed_everything(seed=args.seed)

if __name__=='__main__':
    # telecolab.start('Calibration: resnet50')


    models = [ model[0] for model in list(models_names_dict.values())]
    model_names = [ model[1] for model in list(models_names_dict.values())]
    model_zoos = [ model[2] for model in list(models_names_dict.values())]
    batch_size = args.batchsize
    device = "cuda" if torch.cuda.is_available() else "cpu"


 
    #################################
    #################################
    bgs = args.bgs.split() #['bg1', 'bg2', 'nobg]
    pose = ['roll']*len(bgs) #+ ['yaw']*len(bgs) + ['pitch']*len(bgs)  #in_plane_roll
    POSES = [pose.upper() for pose in args.poses.split()] #["ROLL"] #IN_PLANE_ROLL

    true_class_dict = {'airliner': [AIRLINER], 'barberchair': [BARBERCHAIR], 'cannon':[CANNON], 'fireengine':[FIREENGINE], 'foldingchair':[FOLDINGCHAIR],
     'forklift':[FORKLIFT], 'garbagetruck':[GARBAGETRUCK], 'hammerhead':[HAMMERHEAD], 'jeep':[JEEP], 'mountainbike':[MOUNTAINBIKE],
     'parkbench':[PARKBENCH], 'rockingchair':[ROCKINGCHAIR], 'shoppingcart':[SHOPPINGCART],
     'tablelamp':[TABLELAMP, LAMPSHADE], 'tank':[TANK], 'tractor':[TRACTOR], 'wheelbarrow':[WHEELBARROW] }

    objects = list(true_class_dict.keys())

    #################################
    #################################
    accs_and_confs = {}
    dataroot = args.dataroot
    
  
    for  m in range(len(models)):
 
        # telecolab.send(messages=f'Starting on model number {m+1}/{len(models)}: {models[m]}')

        all_result_path = f'{dataroot}/all_results/{model_names[m]}' #f'gdrive/MyDrive/AMMI GP/newdata/rot_both360/rot_both_all_results/{model_names[m]}' #f'gdrive/MyDrive/AMMI GP/newdata/360/all_results_IN_PLANE_ROLL/{model_names[m]}'  #f'gdrive/MyDrive/AMMI GP/newdata/360/all_results/{model_names[m]}' #path to save the allreslt.pkl file #f'gdrive/MyDrive/AMMI GP/newdata/360/all_results_IN_PLANE_ROLL/{model_names[m]}' 
        if not os.path.exists(all_result_path):
            os.mkdir(all_result_path)

        accs_and_confs[model_names[m]] = {}
        #vit-b = 32 , vit-l=8, en = 8

        # rotboth ->vil_L->8, SWIN_384 -> 16  / the rest 32
        # report = report_writer(os.path.join(data_root_path[0], f'model_result'), obj_name=obj)
        for obj in objects:
            print('current model : ',model_names[m])
            print('✅✅✅✅✅✅✅✅✅✅✅✅✅ object: ', obj)
            true_class = true_class_dict[obj]
       
            if os.path.exists(os.path.join(all_result_path, model_names[m]+'_all_results.pkl')): #if this model's file exits
                print(f'✅✅✅✅✅✅✅✅✅✅✅✅✅ result.pkl file exits: ', os.path.join(all_result_path, model_names[m]+'_all_results.pkl') )
                pickle_file = open(os.path.join(all_result_path, model_names[m]+'_all_results.pkl'), "rb")
                accs_and_confs = pickle.load(pickle_file)
                #Create a place for this object -if it is't already exists-
                print(f'✅✅✅✅✅✅✅✅✅✅✅✅✅ {obj}: ', list(accs_and_confs[model_names[m]].keys()))
                #The file exists but the object does not exist
                if not obj in list(accs_and_confs[model_names[m]].keys()): #if the object doesn't exist in the file -> add the object
                  print(f'✅✅✅✅✅✅✅✅✅✅✅✅✅ object {obj} does not exists')
                  accs_and_confs[model_names[m]][obj] = {POSE:{'bg1':{'logits':[], 'labels':[], 'correct':0}, 'bg2':{'logits':[], 'labels':[], 'correct':0}, 'nobg':{'logits':[], 'labels':[], 'correct':0} } for POSE in POSES }
                else:
                  print(f'✅✅✅✅✅✅✅✅✅✅✅✅✅ object {obj} already exists')

            else:
                print(f'✅✅✅✅✅✅✅✅✅✅✅✅✅ result.pkl file does not exists')

                accs_and_confs[model_names[m]][obj] = {POSE:{'bg1':{'logits':[], 'labels':[], 'correct':0}, 'bg2':{'logits':[], 'labels':[], 'correct':0}, 'nobg':{'logits':[], 'labels':[], 'correct':0} } for POSE in POSES }
            ############################################################################################################################

            data_root_path = [f"{dataroot}/{POSE}/{bg}/{obj}_{POSE}_360" for POSE in POSES for  bg in bgs] # path to load the data from

            savepath = [f"{dataroot}/{POSE}/{bg}/{obj}_{POSE}_360/model_result/{model_names[m]}" for POSE in POSES for  bg in bgs ] # path to save the results
            ############################################################################################################################

            all_labels, all_logits, bg_num, all_correct = [], [], 0, 0
            for i in range(len(data_root_path)):
                print(f'working on object number {i+1}/{len(data_root_path)}')
                P = data_root_path[i].split('/')[6]
                B = data_root_path[i].split('/')[7]

                if not os.path.exists(f'{dataroot}/{P}/{B}/{obj}_{P}_360/model_result'):  #rot_both360
                  os.mkdir(f'{dataroot}/{P}/{B}/{obj}_{P}_360/model_result')
                if not os.path.exists(f'{dataroot}/{P}/{B}/{obj}_{P}_360/model_result/{model_names[m]}'):
                  os.mkdir(f'{dataroot}/{P}/{B}/{obj}_{P}_360/model_result/{model_names[m]}')
        
                data_transform = transforms.Compose( [ 
                          # transforms.Scale(224),
                          transforms.Resize(size=input_size),
                          transforms.CenterCrop(size=(input_size, input_size)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
                
                data = ImagePoseData(os.path.join(data_root_path[i], f'images'), transform=data_transform) #rot_both_images
                mydataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
              ######################################################################
              # run the model and save the log.txt and result.txt files to the savepath
                correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, result, result5, logits, labels =  run_model(models[m], mydataloader,
                                                                                        list(imagenet_classes.values()),savepath[i], 
                                                                                    pose=pose[i], model_name=model_names[m], model_zoo=model_zoos[m],
                                                                                    true_classes=true_class, report=None,device=device)
              #####################################################################
                #Calibration
                accs_and_confs[model_names[m]][obj][POSES[i//3]][bgs[i%3]]= {'logits':logits.tolist(), 'labels':labels.tolist()[0], 'correct':correct}
 
            pickle_file = open(os.path.join(all_result_path, model_names[m]+'_all_results.pkl'), "wb")
            pickle.dump(accs_and_confs, pickle_file) 
        # log = open (os.path.join(os.path.join( all_result_path, model_names[m]+'_all_results.yml' ), "w") #summary table

# telecolab.end()


##################################

import torch
from PIL import Image
from models.CLIP import clip
import os
import numpy as np
from tabulate import tabulate
import yaml
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import re
from utils import sort_alphanumerically

# device = "cuda" if torch.cuda.is_available() else "cpu"

def run_model(model, dataloader, imagenet_classes, savepath,  pose, model_name, model_zoo, true_classes, report, device):
    '''
    Function to run vision models

    params: 
        model: vision model
        dataloader: bs = len(data)
        imagenet_classes: python list
        savepath: path to save the model_log.txt file and model_result.txt file
        pose: rotation axis
        model_name: model name (str)
        model_zoo: ['torchvision', 'modelvshuman', 'clip']
        true_classes: list of true classes for the image [first_true_class_id, second_true_class_id, ...]
    return:
        <model_name>_log.txt: a table contains: top1, top5, correct_t_conf, correct_f_conf, wrong_conf
        <model_name>_result.txt: result [image: 'correct' or 'wrong']
        <model_name>_result5.yml: result_dict5 {image_name: [correct/wrong , ids, probs]}
        <model_name>_top5_file.txt: top5 preds tables
    '''

    logits_list = []
    labels_list = []

    print(f'Running {model_name} on the {imagenet_classes[true_classes[0]].split()} class')
    print(f'POSE: {pose}')

    if not os.path.isdir(savepath):
      os.mkdir(savepath)
    if report != None:
      report.write('\n'+f'Running {model_name} on the {imagenet_classes[true_classes[0]].split()} / {pose} data'+'\n')

    correct_t_conf, correct_f_conf, wrong_conf=[], [], []
    correct5, correct=0,0 #top1, and top5 correct predictions 

    # result_file = open (os.path.join(savepath, model_name+"_result.txt"), "w") #file to save the resut dictionary.

    result_dict = {} #dict to save the either the top1 class is the ture class for each image in the form {(image_name: correct/wrong)}.
    result_dict5 = {}  #dict to save the if the true class is among the top5 preds in the form {image_name: [correct/wrong , ids, probs]}.

    log = open (os.path.join(savepath, model_name+"_log.txt"), "w") #summary table

    top5_tables = open (os.path.join(savepath, model_name+"_top5_tables.txt"), "a") #file to save all the top5 predictions for all the images
    top5_tables.truncate(0) #need '0' when using r+

    print('predicting...')
    for batch, names in dataloader:

      with torch.no_grad():
        if model_zoo in ['torchvision', 'timm', 'ptrnets', 'torchhub', 'pytorch_pretrained_vit']: #torchvision model
          model.eval()
          model = model.to(device) 
          output = model(batch.to(device))
          probs = output.softmax(dim=-1)

        elif model_zoo=="modelvshuman": #modelvshuman
          output = model.forward_batch(batch.to(device))
          output = torch.tensor(output)
          probs = output.softmax(dim=-1)

        elif model_zoo=='clip':
          image_input = batch.to(device) 
          text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in imagenet_classes]).to(device) 
          with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

          # Pick the top 5 most similar labels for the image
          image_features /= image_features.norm(dim=-1, keepdim=True)
          text_features /= text_features.norm(dim=-1, keepdim=True)
          similarity = (100.0 * image_features @ text_features.T)
          output = similarity
          probs = similarity.softmax(dim=-1) #normalized similarity

      ######for calibration
      logits_list.append(output)
      label = torch.tensor([true_classes[0]]*batch.shape[0]).to(device)
      labels_list.append(label)


      values, indices = probs.topk(5) #[bs x 5]
      batch_size = values.shape[0]
      for i in range(batch_size): #For all the <batch_size> images
        #top5
        if len(set(true_classes).intersection(indices[i].tolist())) > 0:    #if true_class in indices[i]
          correct5+=1
          result_dict5[names[i]] = ['correct', indices[i].tolist(), values[i].tolist()]
        else:
          result_dict5[names[i]] = ['wrong', indices[i].tolist(), values[i].tolist()]
        #top1
        if len(set(true_classes).intersection( [indices[i][0].item()] ) ) > 0: #if true_class==indices[i][0]:
          correct+=1
          result_dict[names[i]] = 'correct'
          correct_t_conf.append(values[i][0].item())
        else:
          result_dict[names[i]] = 'wrong'
          wrong_conf.append(values[i][0].item())
          correct_f_conf.append(probs[i][true_classes[0]].item())
        
        #Create table for top5 predictions
        # for j in range(values.shape[0]): #For all the <batch_size> images
        top5_class_index = ['class index']+[indices[i][j] for j in range(len(indices[i]))]
        top5_list = ['class']+[ re.sub(r'^(.{30}).*$', '\g<1>...', imagenet_classes[indices[i][j]]) for j in range(len(indices[i]))] #indices[j]==[5x1]==>One example
        top5_probs = ['probs']+[values[i][j] for j in range(len(values[i]))]
        top5_correct = ['correct']+[len(set(true_classes).intersection([indices[i][j].item()])) > 0 for j in range(len(indices[i]))]  #['correct']+[indices[i][j]==true_classes for j in range(len(indices[i]))]
        tpo5_table = tabulate([top5_class_index, top5_list, top5_probs, top5_correct], headers=[names[i], '1', '2', '3', "4", "5"])
        top5_tables.write('\n'+tpo5_table+'\n')

    num_images=len(dataloader.sampler)
    print(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')
    if report != None:
      report.write(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')

    log.write('\n'+'correct_t_conf'+'\n')
    log.write('\n'+'['+','.join([str(elem) for elem in correct_t_conf] )+']'+'\n')
    log.write('\n'+'wrong_conf'+'\n')
    log.write('\n'+'['+','.join([str(elem) for elem in wrong_conf] )+']'+'\n')
    log.write('\n'+'correct_f_conf'+'\n')
    log.write('\n'+'['+','.join([str(elem) for elem in correct_f_conf] )+']'+'\n')

    correct_t_conf = np.mean(correct_t_conf)
    wrong_conf = np.mean(wrong_conf)
    correct_f_conf = np.mean(correct_f_conf)

    table_data =[[pose, num_images, correct, correct5, 100*correct/num_images, 100*correct5/num_images, correct_t_conf, correct_f_conf, wrong_conf, model_name]]
    table = tabulate(table_data, headers=['pose', 'num_images', 'correct', 'correct5', 'top1_acc', 'top5_acc', "correct_t_conf", "correct_f_conf", "wrong_conf", 'model'])
    print (table)
    if report != None:
       report.write('\n'+table+'\n')
    log.write('\n'+table+'\n')
    log.close()
    # result_file.write('\n'+str(result_dict)+'\n')
    # result_file.close()
    #Save result dict and result5 dict to .yml files
    with open(os.path.join(savepath, model_name+'_result.yml'), 'w') as file:
      yaml_file = yaml.dump(result_dict, file)
    with open(os.path.join(savepath, model_name+'_result5.yml'), 'w') as file:
      yaml_file = yaml.dump(result_dict5, file)
    
    del model
    if report != None:
       report.write('\n ##################################################" \n')

    #for calibration
    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()

    return correct, correct5, np.mean(correct_t_conf), np.mean(correct_f_conf), np.mean(wrong_conf), result_dict, result_dict5, logits, labels






# import torch
# from PIL import Image
# from models.CLIP import clip
# import os
# import numpy as np
# from tabulate import tabulate
# import yaml
# import numpy as np
# import matplotlib.pyplot as plt; plt.rcdefaults()
# import re
# from utils import sort_alphanumerically

# # device = "cuda" if torch.cuda.is_available() else "cpu"

# def run_model(model, dataloader, imagenet_classes, savepath,  pose, model_name, model_zoo, true_classes, report, device):
#     '''
#     Function to run vision models

#     params: 
#         model: vision model
#         dataloader: bs = len(data)
#         imagenet_classes: python list
#         savepath: path to save the model_log.txt file and model_result.txt file
#         pose: rotation axis
#         model_name: model name (str)
#         model_zoo: ['torchvision', 'modelvshuman', 'clip']
#         true_classes: list of true classes for the image [first_true_class_id, second_true_class_id, ...]
#     return:
#         <model_name>_log.txt: a table contains: top1, top5, correct_t_conf, correct_f_conf, wrong_conf
#         <model_name>_result.txt: result [image: 'correct' or 'wrong']
#         <model_name>_result5.yml: result_dict5 {image_name: [correct/wrong , ids, probs]}
#         <model_name>_top5_file.txt: top5 preds tables
#     '''

#     logits_list = []
#     labels_list = []

#     print(f'Running {model_name} on the {imagenet_classes[true_classes[0]].split()} class')
#     print(f'POSE: {pose}')

#     if not os.path.isdir(savepath):
#       os.mkdir(savepath)
#     if report != None:
#       report.write('\n'+f'Running {model_name} on the {imagenet_classes[true_classes[0]].split()} / {pose} data'+'\n')

#     correct_t_conf, correct_f_conf, wrong_conf=[], [], []
#     correct5, correct=0,0 #top1, and top5 correct predictions 

#     # result_file = open (os.path.join(savepath, model_name+"_result.txt"), "w") #file to save the resut dictionary.

#     result_dict = {} #dict to save the either the top1 class is the ture class for each image in the form {(image_name: correct/wrong)}.
#     result_dict5 = {}  #dict to save the if the true class is among the top5 preds in the form {image_name: [correct/wrong , ids, probs]}.

#     log = open (os.path.join(savepath, model_name+"_log.txt"), "w") #summary table

#     top5_tables = open (os.path.join(savepath, model_name+"_top5_tables.txt"), "a") #file to save all the top5 predictions for all the images
#     top5_tables.truncate(0) #need '0' when using r+

#     print('predicting...')
#     for batch, names in dataloader:

#       with torch.no_grad():
#         if model_zoo in ['torchvision', 'timm', 'ptrnets', 'torchhub', 'pytorch_pretrained_vit']: #torchvision model
#           model.eval()
#           model = model.to(device) 
#           output = model(batch.to(device))
#           probs = output.softmax(dim=-1)

#         elif model_zoo=="modelvshuman": #modelvshuman
#           output = model.forward_batch(batch.to(device))
#           output = torch.tensor(output)
#           probs = output.softmax(dim=-1)

#         elif model_zoo=='clip':
#           image_input = batch.to(device) 
#           text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in imagenet_classes]).to(device) 
#           with torch.no_grad():
#             image_features = model.encode_image(image_input)
#             text_features = model.encode_text(text_inputs)

#           # Pick the top 5 most similar labels for the image
#           image_features /= image_features.norm(dim=-1, keepdim=True)
#           text_features /= text_features.norm(dim=-1, keepdim=True)
#           similarity = (100.0 * image_features @ text_features.T)
#           output = similarity
#           probs = similarity.softmax(dim=-1) #normalized similarity

#       ######for calibration
#       logits_list.append(output)
#       label = torch.tensor([true_classes[0]]*batch.shape[0]).to(device)
#       labels_list.append(label)


#       values, indices = probs.topk(5) #[bs x 5]
#       batch_size = values.shape[0]
#       for i in range(batch_size): #For all the <batch_size> images
#         #top5
#         if len(set(true_classes).intersection(indices[i].tolist())) > 0:    #if true_class in indices[i]
#           correct5+=1
#           result_dict5[names[i]] = ['correct', indices[i].tolist(), values[i].tolist()]
#         else:
#           result_dict5[names[i]] = ['wrong', indices[i].tolist(), values[i].tolist()]
#         #top1
#         if len(set(true_classes).intersection( [indices[i][0].item()] ) ) > 0: #if true_class==indices[i][0]:
#           correct+=1
#           result_dict[names[i]] = 'correct'
#           correct_t_conf.append(values[i][0].item())
#         else:
#           result_dict[names[i]] = 'wrong'
#           wrong_conf.append(values[i][0].item())
#           correct_f_conf.append(probs[i][true_classes[0]].item())
        
#         #Create table for top5 predictions
#         # for j in range(values.shape[0]): #For all the <batch_size> images
#         top5_class_index = ['class index']+[indices[i][j] for j in range(len(indices[i]))]
#         top5_list = ['class']+[ re.sub(r'^(.{30}).*$', '\g<1>...', imagenet_classes[indices[i][j]]) for j in range(len(indices[i]))] #indices[j]==[5x1]==>One example
#         top5_probs = ['probs']+[values[i][j] for j in range(len(values[i]))]
#         top5_correct = ['correct']+[len(set(true_classes).intersection([indices[i][j].item()])) > 0 for j in range(len(indices[i]))]  #['correct']+[indices[i][j]==true_classes for j in range(len(indices[i]))]
#         tpo5_table = tabulate([top5_class_index, top5_list, top5_probs, top5_correct], headers=[names[i], '1', '2', '3', "4", "5"])
#         top5_tables.write('\n'+tpo5_table+'\n')

#     num_images=len(dataloader.sampler)
#     print(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')
#     if report != None:
#       report.write(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')

#     log.write('\n'+'correct_t_conf'+'\n')
#     log.write('\n'+'['+','.join([str(elem) for elem in correct_t_conf] )+']'+'\n')
#     log.write('\n'+'wrong_conf'+'\n')
#     log.write('\n'+'['+','.join([str(elem) for elem in wrong_conf] )+']'+'\n')
#     log.write('\n'+'correct_f_conf'+'\n')
#     log.write('\n'+'['+','.join([str(elem) for elem in correct_f_conf] )+']'+'\n')

#     correct_t_conf = np.mean(correct_t_conf)
#     wrong_conf = np.mean(wrong_conf)
#     correct_f_conf = np.mean(correct_f_conf)

#     table_data =[[pose, num_images, correct, correct5, 100*correct/num_images, 100*correct5/num_images, correct_t_conf, correct_f_conf, wrong_conf, model_name]]
#     table = tabulate(table_data, headers=['pose', 'num_images', 'correct', 'correct5', 'top1_acc', 'top5_acc', "correct_t_conf", "correct_f_conf", "wrong_conf", 'model'])
#     print (table)
#     if report != None:
#        report.write('\n'+table+'\n')
#     log.write('\n'+table+'\n')
#     log.close()
#     # result_file.write('\n'+str(result_dict)+'\n')
#     # result_file.close()
#     #Save result dict and result5 dict to .yml files
#     with open(os.path.join(savepath, model_name+'_result.yml'), 'w') as file:
#       yaml_file = yaml.dump(result_dict, file)
#     with open(os.path.join(savepath, model_name+'_result5.yml'), 'w') as file:
#       yaml_file = yaml.dump(result_dict5, file)
    
#     del model
#     if report != None:
#        report.write('\n ##################################################" \n')

#     #for calibration
#     logits = torch.cat(logits_list).cuda()
#     labels = torch.cat(labels_list).cuda()

#     return correct, correct5, np.mean(correct_t_conf), np.mean(correct_f_conf), np.mean(wrong_conf), result_dict, result_dict5, logits, labels





#############


def get_inv2_data_tranforms(model_zoo, model_name, input_size):

    if 'beit_base'==model_zoo:
        beit_base_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
        data_transform= transforms.Compose( [ 
          lambda images: beit_base_feature_extractor(images, return_tensors="pt")['pixel_values'].squeeze(0) ])
        
    elif 'beit_large'==model_zoo:
        beit_large_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
        data_transform= transforms.Compose( [ 
          lambda images: beit_large_feature_extractor(images, return_tensors="pt")['pixel_values'].squeeze(0) ])
        
    elif 'clip_ensemble'==model_zoo or 'clip'==model_zoo:
        data_transform= transforms.Compose([
                              transforms.Resize(input_size, interpolation= Image.BICUBIC),
                              transforms.CenterCrop((input_size,input_size)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                          ])
        
    elif 'Efficientnet_l2_475_noisy_student' == model_name:
          normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          crop_pct = 0.936
          scale_size = int(math.floor(input_size / crop_pct)) 
          data_transform= transforms.Compose([
              transforms.Resize(scale_size, interpolation=Image.BICUBIC),
              transforms.CenterCrop(input_size),
              transforms.ToTensor(),
              normalize,
          ])

    elif 'Efficientnet_b7_600_noisy_student' == model_name:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      crop_pct = 0.949
      scale_size = int(math.floor(input_size / crop_pct)) 
      data_transform= transforms.Compose([
          transforms.Resize(scale_size, interpolation=Image.BICUBIC),
          transforms.CenterCrop(input_size),
          transforms.ToTensor(),
          normalize,
      ])

    elif 'Efficientnet_l2_800_noisy_student' == model_name:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      crop_pct = 0.96
      scale_size = int(math.floor(input_size / crop_pct)) 
      data_transform= transforms.Compose([
          transforms.Resize(scale_size, interpolation=Image.BICUBIC),
          transforms.CenterCrop(input_size),
          transforms.ToTensor(),
          normalize,
      ])

    elif 'mixer' in model_name:
      data_transform = transforms.Compose( [ 
              # transforms.Scale(224),
              transforms.Resize(size=input_size, interpolation=Image.BICUBIC),
              transforms.CenterCrop(size=(input_size, input_size)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
              ])
    elif 'ViT_21k' in model_name:
      data_transform = transforms.Compose([
                        transforms.Resize(input_size, interpolation=Image.BICUBIC),
                        transforms.CenterCrop(size=(input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5),
                    ])
         
    else:
        data_transform = transforms.Compose( [ 
              transforms.Resize(size=int((256/224)*input_size)),
              transforms.CenterCrop(size=(input_size, input_size)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ])
        
    return data_transform




########################################




# import torch
# from PIL import Image
# from models.CLIP import clip
# import os
# import numpy as np
# from tabulate import tabulate
# import yaml
# import numpy as np
# import matplotlib.pyplot as plt; plt.rcdefaults()
# import re
# from utils import sort_alphanumerically

# # device = "cuda" if torch.cuda.is_available() else "cpu"

# def run_model(model, dataloader, imagenet_classes, savepath,  pose, model_name, model_zoo, true_classes, report, device):
#     '''
#     Function to run vision models

#     params: 
#         model: vision model
#         dataloader: bs = len(data)
#         imagenet_classes: python list
#         savepath: path to save the model_log.txt file and model_result.txt file
#         pose: rotation axis
#         model_name: model name (str)
#         model_zoo: ['torchvision', 'modelvshuman', 'clip']
#         true_classes: list of true classes for the image [first_true_class_id, second_true_class_id, ...]
#     return:
#         <model_name>_log.txt: a table contains: top1, top5, ground_truth_true_conf, ground_truth_false_conf, wrong_conf
#         <model_name>_result.txt: result [image: 'correct' or 'wrong']
#         <model_name>_result5.yml: result_dict5 {image_name: [correct/wrong , ids, probs]}
#         <model_name>_top5_file.txt: top5 preds tables
#     '''

#     logits_list = []
#     labels_list = []

#     print(f'Running {model_name} on the {imagenet_classes[true_classes[0]].split()} class')
#     print(f'POSE: {pose}')

#     if not os.path.isdir(savepath):
#       os.mkdir(savepath)
#     if report != None:
#       report.write('\n'+f'Running {model_name} on the {imagenet_classes[true_classes[0]].split()} / {pose} data'+'\n')

#     ground_truth_true_conf, ground_truth_false_conf, wrong_conf=[], [], []
#     correct5, correct=0,0 #top1, and top5 correct predictions 

#     # result_file = open (os.path.join(savepath, model_name+"_result.txt"), "w") #file to save the resut dictionary.

#     result_dict = {} #dict to save the either the top1 class is the ture class for each image in the form {(image_name: correct/wrong)}.
#     result_dict5 = {}  #dict to save the if the true class is among the top5 preds in the form {image_name: [correct/wrong , ids, probs]}.

#     log = open (os.path.join(savepath, model_name+"_log.txt"), "w") #summary table

#     top5_tables = open (os.path.join(savepath, model_name+"_top5_tables.txt"), "a") #file to save all the top5 predictions for all the images
#     top5_tables.truncate(0) #need '0' when using r+

#     print('predicting...')
#     for batch, names in dataloader:

#       with torch.no_grad():
#         if model_zoo in ['torchvision', 'timm', 'ptrnets', 'torchhub', 'pytorch_pretrained_vit']: #torchvision model
#           model.eval()
#           model = model.to(device) 
#           output = model(batch.to(device))
#           probs = output.softmax(dim=-1)

#         elif model_zoo=="modelvshuman": #modelvshuman
#           output = model.forward_batch(batch.to(device))
#           output = torch.tensor(output)
#           probs = output.softmax(dim=-1)

#         elif model_zoo=='clip':
#           image_input = batch.to(device) 
#           text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in imagenet_classes]).to(device) 
#           with torch.no_grad():
#             image_features = model.encode_image(image_input)
#             text_features = model.encode_text(text_inputs)

#           # Pick the top 5 most similar labels for the image
#           image_features /= image_features.norm(dim=-1, keepdim=True)
#           text_features /= text_features.norm(dim=-1, keepdim=True)
#           similarity = (100.0 * image_features @ text_features.T)
#           output = similarity
#           probs = similarity.softmax(dim=-1) #normalized similarity

#       ######for calibration
#       logits_list.append(output)
#       label = torch.tensor([true_classes[0]]*batch.shape[0]).to(device)
#       labels_list.append(label)


#       top5_props, top5_indices = probs.topk(5) #[bs x 5]
#       batch_size = top5_props.shape[0]
#       for i in range(batch_size): #For all the <batch_size> images
#         #top5
#         if len(set(true_classes).intersection(top5_indices[i].tolist())) > 0:    #if true_class in top5_indices[i]
#           correct5+=1
#           result_dict5[names[i]] = ['correct', top5_indices[i].tolist(), top5_props[i].tolist()]
#         else:
#           result_dict5[names[i]] = ['wrong', top5_indices[i].tolist(), top5_props[i].tolist()]
#         #top1
#         if len(set(true_classes).intersection( [top5_indices[i][0].item()] ) ) > 0: #if true_class==top5_indices[i][0]:
#           correct+=1
#           result_dict[names[i]] = 'correct'
#           ground_truth_true_conf.append(top5_props[i][0].item())
#         else:
#           result_dict[names[i]] = 'wrong'
#           wrong_conf.append(top5_props[i][0].item())
#           ground_truth_false_conf.append(probs[i][true_classes[0]].item())
        
#         #Create table for top5 predictions
#         # for j in range(top5_props.shape[0]): #For all the <batch_size> images
#         top5_class_index = ['class index']+[top5_indices[i][j] for j in range(len(top5_indices[i]))]
#         top5_list = ['class']+[ re.sub(r'^(.{30}).*$', '\g<1>...', imagenet_classes[top5_indices[i][j]]) for j in range(len(top5_indices[i]))] #top5_indices[j]==[5x1]==>One example
#         top5_probs = ['probs']+[top5_props[i][j] for j in range(len(top5_props[i]))]
#         top5_correct = ['correct']+[len(set(true_classes).intersection([top5_indices[i][j].item()])) > 0 for j in range(len(top5_indices[i]))]  #['correct']+[top5_indices[i][j]==true_classes for j in range(len(top5_indices[i]))]
#         tpo5_table = tabulate([top5_class_index, top5_list, top5_probs, top5_correct], headers=[names[i], '1', '2', '3', "4", "5"])
#         top5_tables.write('\n'+tpo5_table+'\n')

#     num_images=len(dataloader.sampler)
#     print(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')
#     if report != None:
#       report.write(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')

#     log.write('\n'+'ground_truth_true_conf'+'\n')
#     log.write('\n'+'['+','.join([str(elem) for elem in ground_truth_true_conf] )+']'+'\n')
#     log.write('\n'+'wrong_conf'+'\n')
#     log.write('\n'+'['+','.join([str(elem) for elem in wrong_conf] )+']'+'\n')
#     log.write('\n'+'ground_truth_false_conf'+'\n')
#     log.write('\n'+'['+','.join([str(elem) for elem in ground_truth_false_conf] )+']'+'\n')

#     ground_truth_true_conf = np.mean(ground_truth_true_conf)
#     wrong_conf = np.mean(wrong_conf)
#     ground_truth_false_conf = np.mean(ground_truth_false_conf)

#     table_data =[[pose, num_images, correct, correct5, 100*correct/num_images, 100*correct5/num_images, ground_truth_true_conf, ground_truth_false_conf, wrong_conf, model_name]]
#     table = tabulate(table_data, headers=['pose', 'num_images', 'correct', 'correct5', 'top1_acc', 'top5_acc', "ground_truth_true_conf", "ground_truth_false_conf", "wrong_conf", 'model'])
#     print (table)
#     if report != None:
#        report.write('\n'+table+'\n')
#     log.write('\n'+table+'\n')
#     log.close()
#     # result_file.write('\n'+str(result_dict)+'\n')
#     # result_file.close()
#     #Save result dict and result5 dict to .yml files
#     with open(os.path.join(savepath, model_name+'_result.yml'), 'w') as file:
#       yaml_file = yaml.dump(result_dict, file)
#     with open(os.path.join(savepath, model_name+'_result5.yml'), 'w') as file:
#       yaml_file = yaml.dump(result_dict5, file)
    
#     del model
#     if report != None:
#        report.write('\n ##################################################" \n')

#     #for calibration
#     logits = torch.cat(logits_list).cuda()
#     labels = torch.cat(labels_list).cuda()

#     return correct, correct5, np.mean(ground_truth_true_conf), np.mean(ground_truth_false_conf), np.mean(wrong_conf), result_dict, result_dict5, logits, labels





