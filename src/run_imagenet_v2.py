from datetime import datetime
from torch.utils.data import  DataLoader
from Models.Models import models_names_dict
import torch
import argparse
from utils import seed_everything, report_writer
from dataloader import ImageNetV2Dataset, CO3D_Dataset
import os
import pickle

def run_imagenet_v2(models, batch_size, device, datasetname, exprimentname, models_names,  args,
                    datapath='gdrive/MyDrive/AMMI GP/newdata/imagenetv2-matched-frequency', all_result_path='', testcode=False, co3d_id_map=None):

    print('Starting........')
    writer = report_writer(all_result_path,f'{datasetname}_{exprimentname}') #text file with the name < '{datasetname}_{exprimentname}.txt'> to save the result
    
    writer.write(f'###########################\nTesting on: {datasetname}, {exprimentname}\n')
    if testcode: writer.write('Running code test...'); 
    writer.write(f'\n Experiment: {datetime.now().time().strftime("%H:%M:%S")}, {datetime.now().strftime("%Y-%m-%d")}')
    writer.write( '\n Testing with Batch Size={batch_size}')
    writer.write(f'\n Testing the model with cropping: {args.crop}')

    models_accs = {}
    for  m in range(len(models)):
        print(f'model {m+1}/{len(models)}: {models_names[m]}')
        
        model_name = model_names[m]
        print('getting the model')
        model = models[m](device)
        if not args.crop: 
          writer.write('\nTesting without cropping the image')
          model.crop_pct = 1.0
        model.set_transforms()

        print('Model is ready')
        correct1 = 0
        correct5 = 0
        total = 0
        
        #get the dataset
        if datasetname=='imagenetv2':
            dataset = ImageNetV2Dataset(root_dir=datapath, transform=model.transform) 
        elif datasetname=='co3d':
            dataset =  CO3D_Dataset(root_dir=datapath, transform=model.transform, co3d_id_map=co3d_id_map) 

        #dataloader
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        print('Testing........')
        with torch.no_grad():
            for batch, target in test_loader:
                # print('batch')
                target = target.to(device)
                output = model(batch.to(device))
                _, predicted = output.topk(5, 1, True, True)
                predicted = predicted.t()
                
                correct = predicted.eq(target.view(1, -1).expand_as(predicted))
                correct1 += correct[:1].reshape(-1).float().sum(0)
                correct5 += correct[:5].reshape(-1).float().sum(0)
                total += target.size(0)
                
                if testcode: #use one batch only just to test the code
                  break

        #delete the model to free memory space
        del model
        #compute the accuracy
        accuracy1 = (correct1 / total)*100
        accuracy5 = (correct5 / total)*100

        writer.write(f"'{model_name}': 'top1': {accuracy1}, 'top5': {accuracy5} \n")

        models_accs[model_name] = {'top1':accuracy1, 'top5':accuracy5}
        
    pickle_file = open(os.path.join(all_result_path, f'{datasetname}_{exprimentname}_results.pkl'), "wb")
    pickle.dump(models_accs, pickle_file) 



parser = argparse.ArgumentParser(description='imagenetv2-matched-frequency')

parser.add_argument('--dataroot', default = '../../data/imagenetv2-matched-frequency/data', type=str)
parser.add_argument('--modelslist', default = 'all', type=str)
parser.add_argument('--allresultpath', default = '../../data/imagenetv2-matched-frequency/all_results', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batchsize', default=16, type=int)
parser.add_argument('--testcode', default=False, action="store_true")
parser.add_argument('--crop', default=False, action="store_true")
parser.add_argument('--datasetname', default='imagenetv2', type=str)
parser.add_argument('--exprimentname', default='expriment', type=str)

args=parser.parse_args()
seed_everything(seed=args.seed)

if __name__=='__main__':

    BANANA=954; MICROWAVE=651; HOTDOG=934; TOILET=861; TOASTER=859; BROCCOLI= 937;  PARKINGMETER=704;  PARKBENCH=703; MOUNTAINBIKE = 671;  MOTORSCOOTER = 670;


    datasetname = args.datasetname
    if datasetname=='co3d':
        CO3D_CLASSES =  {'banana' : BANANA, 'microwave': MICROWAVE, 'hotdog':HOTDOG, 'toilet': TOILET, 'toaster':TOASTER, 'broccoli':BROCCOLI, 
        'parkingmeter': PARKINGMETER, 'parkbench': PARKBENCH, 'mountainbike':MOUNTAINBIKE, 'motorscooter': MOTORSCOOTER } 

        #We will use all the IN classes, so we need to map the idx for the dataloader
        co3d_id_map = {}
        for folder in os.listdir(args.dataroot):
          co3d_id_map[folder.lower()] = CO3D_CLASSES[folder.lower()]
    else:
        co3d_id_map=None
        
    if args.modelslist != 'all':
        models_names_dict = {model_name:models_names_dict[model_name] for model_name in args.modelslist.split()}

    
    models = [ model[0] for model in list(models_names_dict.values())]
    model_names = [ model[1] for model in list(models_names_dict.values())]
    model_zoos = [ model[2] for model in list(models_names_dict.values())]
    # input_sizes = [ model[3] for model in list(models_names_dict.values())]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running on {device}')

    datapath = args.dataroot #'../../CO3D_subset'
    all_result_path = args.allresultpath
    batch_size = args.batchsize
    testcode = args.testcode
    exprimentname = args.exprimentname

    os.makedirs(all_result_path, exist_ok=True)

    run_imagenet_v2(models, batch_size, device, datasetname, exprimentname, model_names, datapath=datapath, all_result_path=all_result_path, testcode=testcode, co3d_id_map=co3d_id_map, args=args)





