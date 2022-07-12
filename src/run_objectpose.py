import torch
from torch.utils.data import  DataLoader
import os
import pickle
from Models.Models import models_names_dict
from dataloader import ObjectPoseDataset
from run_model import run_model
import argparse
from imagenet_classes import imagenet_classes
from utils import seed_everything
from utils import report_writer
from datetime import datetime


AIRLINER=404;  BARBERCHAIR = 423; CANNON=471; FIREENGINE=555; FOLDINGCHAIR=559; FORKLIFT=561;
GARBAGETRUCK = 569; HAMMERHEAD=4; JEEP=609; MOUNTAINBIKE = 671; PARKBENCH=703; ROCKINGCHAIR=765; SHOPPINGCART=791;
TABLELAMP=846; LAMPSHADE=619; TANK = 847;  TRACTOR = 866; WHEELBARROW = 428; 

parser = argparse.ArgumentParser(description='ObjectPose')

parser.add_argument('--dataroot', default = '../../data/360/rot_rand_roll/images', type=str, help='Path to ObjectPose data.')
parser.add_argument('--saveroot', default = '../../data/360/rot_rand_roll/images', type=str, help='Path to ObjectPose data.')
parser.add_argument('--bgs', default = 'bg1 bg2 nobg', type=str, help='The backgrounds.')
parser.add_argument('--poses', default='ROLL YAW PITCH', type=str, help='The rotation axes.')
parser.add_argument('--modelslist', default = 'all', type=str)
parser.add_argument('--experimentname', default = 'experiment', type=str)
parser.add_argument('--batchsize', default=16, type=int)
parser.add_argument('--savealllogits', default=False, action="store_true")
parser.add_argument('--datasetname', default='obejctpose', type=str)
parser.add_argument('--crop', default=False, action="store_true")
parser.add_argument('--testcode', default=False, action="store_true")
parser.add_argument('--seed', default=42, type=int)

args=parser.parse_args()
seed_everything(seed=args.seed)

if __name__=='__main__':

    if args.modelslist != 'all':
        models_names_dict = {model_name:models_names_dict[model_name] for model_name in args.modelslist.split()}

    models = [ model[0] for model in list(models_names_dict.values())]
    model_names = [ model[1] for model in list(models_names_dict.values())]
    model_zoos = [ model[2] for model in list(models_names_dict.values())]
    batch_size = args.batchsize
    dataroot = args.dataroot
    saveroot = args.saveroot
    testcode = args.testcode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(saveroot, exist_ok=True)
    writer = report_writer(saveroot, f'logs_{args.experimentname}') #text file with the name < '{datasetname}_{exprimentname}.txt'> to save the result
    writer.write(f'\nRunning on {device} device')
    writer.write(f'Testing the model with cropping: {args.crop}')
 
    #################################
    #################################
    bgs = args.bgs.split() #['bg1', 'bg2', 'nobg]
    pose = [pose.lower() for pose in args.poses.split() for _ in range(len(args.bgs.split()))] #['roll']*len(bgs) #+ ['yaw']*len(bgs) + ['pitch']*len(bgs)  #in_plane_roll
    POSES = [pose.upper() for pose in args.poses.split()] #["ROLL"] #IN_PLANE_ROLL

    true_class_dict =  {'airliner': [AIRLINER], 'barberchair': [BARBERCHAIR], 'cannon':[CANNON], 'fireengine':[FIREENGINE], 'foldingchair':[FOLDINGCHAIR],
     'forklift':[FORKLIFT], 'garbagetruck':[GARBAGETRUCK], 'hammerhead':[HAMMERHEAD], 'jeep':[JEEP], 'mountainbike':[MOUNTAINBIKE],
     'parkbench':[PARKBENCH], 'rockingchair':[ROCKINGCHAIR], 'shoppingcart':[SHOPPINGCART],
     'tablelamp':[TABLELAMP, LAMPSHADE], 'tank':[TANK], 'tractor':[TRACTOR], 'wheelbarrow':[WHEELBARROW] }

    objects = list(true_class_dict.keys())

    #################################
    #################################
    #Python dictionary to hold all the logits
    accs_and_confs = {}

    writer.write(f'\n###########################\nTesting on: ObjectPose, {args.experimentname}\n')
    if testcode: writer.write(f'\nRunning code test...'); 
    writer.write(f'\n Experiment: {datetime.now().time().strftime("%H:%M:%S")}, {datetime.now().strftime("%Y-%m-%d")}')
    writer.write(f'\n Testing with Batch Size={batch_size}\n')
    
    for  m in range(len(models)):
        
        correct_sum, correct5_sum, total_num_images = 0, 0, 0
 
        model_name = model_names[m]
        model = models[m](device)
        if not args.crop: 
          writer.write('\ntesting without cropping the image')
          model.crop_pct = 1.0
        model.set_transforms()

        #Python dictionary to hold all the logits
        accs_and_confs[model_name] = {}

        for obj in objects:
            writer.write(f'\n✅✅✅ current model {m+1}/{len(models)}: {model_name}')
            writer.write(f'\n✅✅✅ object: {obj}')

            true_class = true_class_dict[obj]
            
            if args.savealllogits:  
                #Create folder to Save all the predicted logits in the <logits_result_path> path as .pkl file   
                logits_result_path = f'{saveroot}/logits_result/{model_name}' #f'gdrive/MyDrive/AMMI GP/newdata/rot_both360/rot_both_all_results/{model_name}' #f'gdrive/MyDrive/AMMI GP/newdata/360/all_results_IN_PLANE_ROLL/{model_name}'  #f'gdrive/MyDrive/AMMI GP/newdata/360/logits_result/{model_name}' #path to save the allreslt.pkl file #f'gdrive/MyDrive/AMMI GP/newdata/360/all_results_IN_PLANE_ROLL/{model_name}' 
                os.makedirs(logits_result_path, exist_ok=True)  
                
                #check if the all_result.pkl file already exist from the previous run for this model
                if os.path.exists(os.path.join(logits_result_path, model_name+'_all_results.pkl')): #if this model's file exits
                    writer.write(f'\n✅✅✅ result.pkl file exits: ', os.path.join(logits_result_path, model_name+'_all_results.pkl') )
                    pickle_file = open(os.path.join(logits_result_path, model_name+'_all_results.pkl'), "rb")
                    accs_and_confs = pickle.load(pickle_file)
                    #Create a place for this object -if it doesn't already exists-
                    #The file exists but the object does not exist
                    if not obj in list(accs_and_confs[model_name].keys()): #if the object doesn't exist in the file -> add the object
                      writer.write(f'\n✅✅✅ object {obj} does not exists')
                      accs_and_confs[model_name][obj] = {POSE:{'bg1':{'logits':[], 'labels':[], 'correct':0}, 'bg2':{'logits':[], 'labels':[], 'correct':0}, 'nobg':{'logits':[], 'labels':[], 'correct':0} } for POSE in POSES }
                    else:
                      writer.write(f'\n✅✅✅ object {obj} all_result.pkl already exists')

                else:
                    writer.write(f'\n✅✅✅ object {obj} all_result.pkl file does not exists, creating ')

                    accs_and_confs[model_name][obj] = {POSE:{'bg1':{'logits':[], 'labels':[], 'correct':0}, 'bg2':{'logits':[], 'labels':[], 'correct':0}, 'nobg':{'logits':[], 'labels':[], 'correct':0} } for POSE in POSES }
                ############################################################################################################################
          
            if args.datasetname in ['objectpose_3axes_rot', 'objectpose_scale_and_rot', 'objectpose_bg_rot', 'objectpose_scaling']:
              data_root_path = [f"{dataroot}/{obj}_360"]  # path to load the data from
              savepath = [f"{saveroot}/{obj}_360/model_result/{model_name}"] # path to save the result
                      
            else:
              data_root_path = [f"{dataroot}/{POSE}/{bg}/{obj}_{POSE}_360/images" for POSE in POSES for  bg in bgs] # path to load the data from
              savepath = [f"{saveroot}/{POSE}/{bg}/{obj}_{POSE}_360/model_result/{model_name}" for POSE in POSES for  bg in bgs ] # path to save the result

            ############################################################################################################################

            all_labels, all_logits, bg_num, all_correct = [], [], 0, 0
            for i in range(len(data_root_path)):
                os.makedirs(savepath[i], exist_ok=True)
                writer.write(f'\nworking on object number {i+1}/{len(data_root_path)}')

                
                data = ObjectPoseDataset(os.path.join(data_root_path[i]), transform=model.transform) #rot_both_images
                mydataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
              ######################################################################
              # run the model and save the log.txt and result.txt files to the savepath
                correct, correct5, num_images, correct_t_conf, wrong_conf, result5, logits, labels =  run_model(model, mydataloader,
                                                                                        list(imagenet_classes.values()),savepath[i], 
                                                                                    pose=pose[i], model_name=model_name, 
                                                                                    true_classes=true_class, test_code=args.testcode, save_all_logits = args.savealllogits,
                                                                                    writer=writer, report=None,device=device)

                correct_sum+=correct
                correct5_sum+=correct5
                total_num_images+=num_images
              #####################################################################
                #Calibration
                if args.savealllogits:
                  P =  pose[i].upper()  #{0,1,2}=>0 | {3,4,5}=>1 | {6,7,8}=>2 #data_root_path[i].split('/')[6]
                  B = bgs[i%len(bgs)]   #{0,3,6}=>0 | {1,4,7}=>1 | {2,5,8}=>2   #data_root_path[i].split('/')[7]
                  accs_and_confs[model_name][obj][P][B]= {'logits':logits.tolist(), 'labels':labels.tolist()[0], 'correct':correct}
            if args.savealllogits:
              pickle_file = open(os.path.join(logits_result_path, model_name+'_all_results.pkl'), "wb")
              pickle.dump(accs_and_confs, pickle_file) 

        os.makedirs(os.path.join(saveroot, 'average_acc'), exist_ok=True)
        pickle_file = open(os.path.join(saveroot, 'average_acc', model_name+'_average_acc.pkl'), "wb")
        pickle.dump({'correct_sum': correct_sum, 'correct5_sum':correct5_sum, 'total_num_images':total_num_images}, pickle_file)  


    writer.write(f'\n DONE: {datetime.now().time().strftime("%H:%M:%S")}, {datetime.now().strftime("%Y-%m-%d")} \n')
  