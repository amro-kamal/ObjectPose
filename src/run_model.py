import torch
import os
import numpy as np
from tabulate import tabulate
import yaml
import numpy as np


def run_model(model, dataloader, imagenet_classes, savepath,  pose, model_name, true_classes, report, device, writer, test_code=False, save_all_logits=False):

    '''
    Function to run vision models

    params: 
        model: vision model (Model)
        dataloader: ObjectPose dataloader
        imagenet_classes: python list
        savepath: path to save the model_log.txt file and model_result.txt file (str)
        pose: rotation axis (str)
        model_name: model name (str)
        true_classes: list of true classes for the image [first_true_class_id, second_true_class_id, ...]
    return:
        <model_name>_log.txt: a table contains: top1, top5, ground_truth_true_conf, ground_truth_false_conf, wrong_conf
              # pose      num_images    correct    correct5    top1_acc    top5_acc   ground_truth_true_conf    ground_truth_false_conf    wrong_conf      model
              # ------  ------------  ---------  ----------  ----------  ----------  ------------------------   ------------------------  ------------  ---------------------
              # roll             180         46          87     25.5556     48.3333          0.761681                   0.0293126           0.356618     vit_small_patch16_224

        <model_name>_result5.yml: result_dict5 
             {'image_name': {'top1':'correct'/'wrong', 'top5':'correct'/'wrong', 'top5_indices':xxx, 'top5_props':xxx}}

        <model_name>_top5_file.txt: top5 preds tables
    '''

    writer.write(f'\nRunning {model_name} on the {imagenet_classes[true_classes[0]].split()} class')
    writer.write(f'\nPOSE: {pose}')

    if not os.path.isdir(savepath):
      os.mkdir(savepath)
    if report != None:
      report.write(f'\nRunning {model_name} on the {imagenet_classes[true_classes[0]].split()} / {pose} data\n')


    result_dict5 = {}  #dict to save all the top5 predictions in the form {image_name: [correct/wrong, top5 correct/top5 wrong , top5 ids, top5 probs]}. Will be saved as .yaml file

    #The log file is a summary table:
    # pose      num_images    correct    correct5    top1_acc    top5_acc   ground_truth_true_conf    ground_truth_false_conf    wrong_conf      model
    # ------  ------------  ---------  ----------  ----------  ----------  ------------------------   ------------------------  ------------  ---------------------
    # roll             180         46          87     25.5556     48.3333          0.761681                   0.0293126           0.356618     vit_small_patch16_224
    log = open (os.path.join(savepath, model_name+"_log.txt"), "w") 


    logits_list = []
    labels_list = []
    ground_truth_true_conf, wrong_conf=[], [] #prbabilities
    correct5, correct=0,0 #top1, and top5 correct predictions 
    correct_list, correct5_list=[], []
    top5_indices, top5_props = [], []
    images_names = []

    writer.write(f'\npredicting...')
    for batch, batch_names in dataloader:

      with torch.no_grad():
          # target = target.to(device)
          output = model(batch.to(device))
          probs = output.softmax(dim=-1)

      ######for calibration
      if save_all_logits:
        logits_list.append(output)
        label = torch.tensor([true_classes[0]]*batch.shape[0]).to(device)
        labels_list.append(label)

      batch_top5_props, batch_top5_indices = probs.topk(5) #[batchsize x 5]
      batch_top5_props= batch_top5_props.cpu()
      batch_top5_indices = batch_top5_indices.cpu()
      batch_size = batch_top5_props.shape[0]
      batch_correct_list, batch_correct5_list = torch.tensor([False]*batch_size), torch.tensor([False]*batch_size)
      #if the images has more than one correct classs
      for ture_class in true_classes:
        targets = torch.tensor([ture_class]*batch_size).reshape(batch_size,1)
        batch_correct_list = torch.logical_or(batch_correct_list, batch_top5_indices.eq(targets)[:,0])
        batch_correct5_list = torch.logical_or(batch_correct5_list, torch.any(batch_top5_indices.eq(targets), dim=1))


      correct += torch.sum(batch_correct_list).item()
      correct5 += torch.sum(batch_correct5_list).item()
      correct_list += batch_correct_list.tolist()
      correct5_list += batch_correct5_list.tolist()
      
      top5_indices += batch_top5_indices.tolist()
      top5_props += batch_top5_props.tolist()
      images_names += list(batch_names)

      ground_truth_true_conf += batch_top5_props[batch_correct_list.tolist()][:,0].tolist()
      wrong_conf += batch_top5_props[~torch.tensor(batch_correct_list)][:,0].tolist()
      
      if test_code:
        break
    
    num_images=len(dataloader.sampler)
    result_dict5={'images_names':images_names,'correct_list': correct_list, 'correct5_list': correct5_list, 'correct': correct, 'correct5': correct5,
                  'top5_indices': top5_indices, 'top5_props': top5_props, 'ground_truth_true_conf':ground_truth_true_conf, 'wrong_conf':wrong_conf, 'num_images':num_images}


    
    writer.write(f'\ntop1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')
    if report != None:
      report.write(f'\ntop1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')

    ##########  log file #########################
    log.write(f'\n'+'ground_truth_true_conf'+'\n')
    log.write(f'\n'+'['+','.join([str(elem) for elem in ground_truth_true_conf] )+']'+'\n')
    log.write(f'\n'+'wrong_conf'+'\n')
    log.write(f'\n'+'['+','.join([str(elem) for elem in wrong_conf] )+']'+'\n')

    mean_ground_truth_true_conf = np.mean(ground_truth_true_conf)
    mean_wrong_conf = np.mean(wrong_conf)

    table_data =[[pose, num_images, correct, correct5, 100*correct/num_images, 100*correct5/num_images, mean_ground_truth_true_conf, mean_wrong_conf, model_name]]
    table = tabulate(table_data, headers=['pose', 'num_images', 'correct', 'correct5', 'top1_acc', 'top5_acc', "mean_ground_truth_true_conf", "mean_wrong_conf", 'model'])
    writer.write (f'\n'+table+'\n')
    log.write(f'\n'+table+'\n')
    log.close()
    ##########  log file #########################

    if report != None:
       report.write(f'\n'+table+'\n')

    with open(os.path.join(savepath, model_name+'_result5.yml'), 'w') as file:
      yaml_file = yaml.dump(result_dict5, file)
    
    del model
    if report != None:
       report.write(f'\n ##################################################" \n')

    #for calibration
    if save_all_logits:
      logits_list = torch.cat(logits_list)
      labels_list = torch.cat(labels_list)

    return correct, correct5, num_images, mean_ground_truth_true_conf, mean_wrong_conf, result_dict5, logits_list, labels_list

