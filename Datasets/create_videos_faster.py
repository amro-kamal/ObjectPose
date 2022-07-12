import matplotlib; matplotlib.use('agg')

import torch
from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import numpy as np
from tabulate import tabulate
import yaml
from PIL import ImageDraw
from strike_utils import imagenet_classes
import cv2
import numpy as np
import glob
import re
from strike_utils import sort_alphanumerically
import matplotlib.pyplot as plt; plt.rcdefaults()
import shutil
import time
import copy
import PIL
#TODO
#1- ImagePoseData class

#To delete the .DS_Store file (needed for MacOS ONLY) use the command: find . -name '.#DS_Store' -type f -delete

def clear_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

############################# ############################# #############################
############################# ############################# #############################
############################# ############################# #############################
def rename_img(model_name, outputpath,names_file, names_file5, save_path,pose='pose', correct_class=609):
    '''
    save_path = "data/360/ROLL/bg1/airliner_ROLL_360"
    image_path = "data/360/ROLL/bg1/airliner_ROLL_360/images_lr600"

    if os.path.isdir(os.path.join(data_root_path[i], 'images_lr600')):
        data_path = os.path.join(data_root_path[i], 'images_lr600')
    else:
        data_path = os.path.join(data_root_path[i], 'images')

    1-Loads the images from images_path and rename them according to the clasifications (correct/wrong) from the names_file.yml file
    then save them to the save_path.
    2-Creates top5 barcharts from the names_file5.yml file.

    renamed_images_top1
    renamed_images_top5
    barplots_top1
    barplots_top5

    '''
        
    if os.path.isdir(os.path.join(save_path, 'images_lr600')):
        images_path = os.path.join(save_path, 'images_lr600')
        print('Using 600x600 images')
    else:
        images_path = os.path.join(save_path, 'images')
        print('Using 299x299 images')
    print('Images will be saved to,',os.path.join(save_path, f'renamed_images_top'))

 
    clear_folder(os.path.join(save_path, f'top1_joined_renamed_images'))
    clear_folder(os.path.join(save_path, f'top5_joined_renamed_images'))
    clear_folder(os.path.join(save_path, f'barplots'))

    k=0
    with open(names_file5, "r") as ymlfile:
        result5 = yaml.load(ymlfile)
    with open(names_file, "r") as ymlfile:
        result = yaml.load(ymlfile)

    for (image_name, classification) in result.items():                                    
        classification_top5 = result5[image_name][0]
        pred_class = imagenet_classes[result5[image_name][1][0]].split(',')[0]
        if k%100==0:
          print(f'image num {k} ...')
        k+=1

        ############# add tags ###############
        ############# add tags ###############

        image = Image.open(os.path.join(images_path, image_name)).resize((600,600))
        image_top5 = copy.deepcopy(image)
        correct = Image.open('icons/correct.png')
        correct = correct.resize((90,90))
        wrong = Image.open('icons/wrong.png')
        wrong = wrong.resize((90,90))

        start1 = time.time()

        mask_im = Image.new("L", correct.size, 0)
        draw = ImageDraw.Draw(mask_im)
        draw.ellipse((5, 5, 85, 85), fill=255)
        if classification == 'correct':
          image.paste(correct, (280, 10),mask_im)
        elif classification == 'wrong':
          image.paste(wrong, (280, 10),mask_im)
        if classification_top5 == 'correct':
          image_top5.paste(correct, (280, 10),mask_im)
        elif classification_top5 == 'wrong':
          image_top5.paste(wrong, (280, 10),mask_im)

        # add the predicted class
        font = ImageFont.truetype("JosefinSans-Bold.ttf", 70)

        d = ImageDraw.Draw(image)
        # d.text((10,65), pred_class , font=font, fill=(0,0,0))

        # if result5[image_name][1][0] in correct_class:
        #   image.paste(correct, (280, 10),mask_im)
        # else :
        #   image.paste(wrong, (280, 10),mask_im)

        # if correct_class[0] in result5[image_name][1] or   correct_class[1] in result5[image_name][1]:
        #   image_top5.paste(correct, (280, 10),mask_im)
        # else :
        #   image_top5.paste(wrong, (280, 10),mask_im)
        # image.save(os.path.join(save_path, 'renamed_images_top1', new_name))

        ############# new name ###############
        ############# new name ###############
        class_name = image_name.split('.')[0].split('_')[0]
        if pose=='yaw' or pose=='pitch' or pose=='roll':
            p1 = image_name.split('.')[0].split('_')[-1]
            p2 = 0
            new_name = '_'.join([class_name, model_name, pose, p1, pred_class, classification+'.'+'png']) #image_name.split('.')[1]])
            new_name_top5 = '_'.join([class_name, model_name, pose, p1, pred_class, classification_top5+'.'+'png']) #image_name.split('.')[1]])

        else:
            p1=image_name.split('.')[0].split('_')[-2]
            p2=image_name.split('.')[0].split('_')[-1]
            new_name = '_'.join([class_name, model_name, pose, p1, p2, pred_class, classification+'.'+'png']) #mage_name.split('.')[1]])
            new_name_top5 = '_'.join([class_name, model_name, pose, p1, p2, pred_class, classification_top5+'.'+'png']) #image_name.split('.')[1]])
        

        if not os.path.exists(os.path.join(save_path, 'renamed_images_top1')):
            os.mkdir(os.path.join(save_path, 'renamed_images_top1'))
        
        if not os.path.exists(os.path.join(save_path, 'renamed_images_top1', model_name)):
            os.mkdir(os.path.join(save_path, 'renamed_images_top1', model_name))

            
        image.save(os.path.join(save_path, 'renamed_images_top1', model_name, new_name))
        ############# bar plot ###############
        ############# bar plot ###############
        indices, values = result5[image_name][1], result5[image_name][2] #result5 = {image_name: [correct/wrong , indices, probs]}
        objects = tuple([imagenet_classes[indices[c]].split(',')[0] for c in range(len(indices))][::-1])
        color = ['green' if (indices[c] in correct_class) else 'cyan' for c in range(len(indices))]
        if color[0]=='cyan': color[0]='red'
        green = False
        for i, c in enumerate(color):
          if c=='green' and green==True:
            color[i] = 'cyan'
          elif c=='green' and green==False:
            green = True
        color = color[::-1]
        
        probs = values[::-1]
        y_pos = np.arange(len(objects))
        plots = plt.barh(y_pos, probs, align='center', alpha=0.5, color=color)

        plt.yticks(y_pos, objects, fontsize=16)
        plt.xlabel('probability', fontsize=16)
        # plt.ylabel('classes')
        plt.xlim(0, 1)
        plt.title(f'{model_name}', fontsize=16)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5,alpha=0.8)
        for bar in plots.patches:
            plt.text(bar.get_width()+0.01, bar.get_y()+0.4,
                    str(round((bar.get_width()), 2))+'%',
                    fontsize=9, fontweight='bold',
                    color='grey')
        # plt.savefig(os.path.join(save_path, f'barplots', new_name), bbox_inches='tight', dpi=500)
        # plt.show()

        # barplot_img = Image.open(os.path.join(save_path, f'barplots', new_name)).resize((600, 600))
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        barplot_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).resize((600, 600))
        plt.close()

        img2 = Image.new("RGB", (600*2, 600), "white")
        img2.paste(image, (0, 0))  
        img2.paste(barplot_img, (600, 0))
        img2.save(os.path.join(save_path, f'top1_joined_renamed_images', new_name))
      
        img2 = Image.new("RGB", (600*2, 600), "white")
        img2.paste(image_top5, (0, 0))  
        img2.paste(barplot_img, (600, 0))
        img2.save(os.path.join(save_path, f'top5_joined_renamed_images', new_name_top5))      

    print('images saved at: ', os.path.join(save_path, f'renamed_images_top'))
   
    start1 = time.time()
    img_array=[]
    for filename in sort_alphanumerically(glob.glob(os.path.join(save_path,f'top1_joined_renamed_images/*.png'))):
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    start2 = time.time()
    print('img array time: ', start2 - start1)
    print('saving  the video')

    out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_video_top1.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 6, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_fast_video_top1.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 9, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    ##top5 videos
    start1 = time.time()
    img_array=[]
    for filename in sort_alphanumerically(glob.glob(os.path.join(save_path,f'top5_joined_renamed_images/*.png'))):
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    start2 = time.time()
    print('img array time: ', start2 - start1)
    print('saving  the video')

    out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_video_top5.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 7, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_fast_video_top5.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # models_names = {'bit':'BiTM-resnetv2-152x2', 'swsl':'ResNeXt101-32x16d-swsl', 'simclr':'simclr-resnet50x1', 'vit':'vit-large-patch16-224'}
    # model_names = ['BiTM-resnetv2-152x2', 'Clip-ViT-B-32','resnet50', 'resnet152','ResNeXt101-32x16d-swsl', 'simclr-resnet50x1', 'vit-large-patch16-224']

####################
    # model_names = ['resnet50', 'resnet152', 'simclr-resnet50x1']
    model_names = ['Efficientnet_l2_475_noisy_student'] #['Beit_L16_224']  # ['BiTM_resnetv2_152x2_448']#
    jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
    assault_rifle=413; whiteshark=2; tigershark=3; cannon=471; mug=504; keyboard=508; tablelamp=846; lampshade=619;
    containership = 510; cup=968; shoppingcart=791; tractor=866; hammerhead=4; rockingchair=765
    mountainbike = 671;
    AIRLINER=404;  BARBERCHAIR = 423; CANNON=471; FIREENGINE=555; FOLDINGCHAIR=559; FORKLIFT=561;
    GARBAGETRUCK = 569; HAMMERHEAD=4; JEEP=609; MOUNTAINBIKE = 671; PARKBENCH=703; ROCKINGCHAIR=765; SHOPPINGCART=791;
    TABLELAMP=846; LAMPSHADE=619; TANK = 847;  TRACTOR = 866; WHEELBARROW = 428; 
    true_class_dict = { 'forklift':[FORKLIFT] }
    # {'airliner': [AIRLINER], 'barberchair': [BARBERCHAIR], 'cannon':[CANNON], 'fireengine':[FIREENGINE], 'foldingchair':[FOLDINGCHAIR],
    #  'forklift':[FORKLIFT], 'garbagetruck':[GARBAGETRUCK], 'hammerhead':[HAMMERHEAD], 'jeep':[JEEP], 'mountainbike':[MOUNTAINBIKE],
    #  'parkbench':[PARKBENCH], 'rockingchair':[ROCKINGCHAIR], 'shoppingcart':[SHOPPINGCART],
    #  'tablelamp':[TABLELAMP, LAMPSHADE], 'tank':[TANK], 'tractor':[TRACTOR], 'wheelbarrow':[WHEELBARROW] }
####################
    # loader, jeep , lamp
    for name, c in true_class_dict.items():
        print(name)
        correct_class = [c]*3 
        obj = name
        pose = ['roll'] #['roll','roll','roll']
        bgs = ['bg1'] #['bg1','bg2','nobg']
        POSE = 'ROLL'

    ####################

        # savepath  = [f"data/rot_bg360/{POSE}/{bg}/{obj}_{POSE}_360/rot_bg_model_result" for bg in bgs]
        # data_root_path = [f"data/rot_bg360/{POSE}/{bg}/{obj}_{POSE}_360/" for bg in bgs]

        # data_root_path  = [f"data/360/{POSE}/{bg}/{obj}_{POSE}_360/" for bg in bgs]
        # savepath = [f"data/360/{POSE} result/result/{bg}/{obj}_{POSE}_360/" for bg in bgs] 

        savepath = [f"newdata/360/{POSE}/{bg}/{obj}_{POSE}_360/model_result" for bg in bgs] 
        data_root_path = [f"newdata/360/{POSE}/{bg}/{obj}_{POSE}_360" for bg in bgs]

        # ROLLPITCH
        # savepath = [f"newdata/360/{POSE}/result/{bg}/{obj}_{POSE}_360/" for bg in bgs] 
        # data_root_path = [f"newdata/360/{POSE}/{bg}/{obj}_{POSE}_360" for bg in bgs]


    ######################################################################

        for m in range(len(model_names)):
            print('current model : ',model_names[m])
            for i in range(len(savepath)):
                start = time.time()

                print(f'working on object number {i+1}/{len(savepath)}')

                # rename the images with the classifications 

                rename_img(model_name=model_names[m], outputpath=savepath[i],
                            names_file=os.path.join(savepath[i], model_names[m],model_names[m]+'_result.yml'), names_file5=os.path.join(savepath[i],model_names[m], model_names[m]+'_result5.yml'), 
                            save_path=data_root_path[i], pose=pose[i], correct_class=correct_class[i])

                print('total time per bg: ',time.time()-start)

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# EN: mb, roll, bg1 | jeep, pitch, bg2 | rc, p, bg2 | fc, p, bg2 | tank, y, bg1 | tactor, y, bg2 | bc, y, bg1
# Beit: mb,p,bg1 | fl,r,bg1 | tl,p,bg1 | j, r,bg1 | rc,p,bg1 | a,r,bg1

# import matplotlib; matplotlib.use('agg')
# import torch
# from PIL import Image, ImageFont, ImageDraw
# from torch.utils.data import Dataset, DataLoader
# import os
# from torchvision import transforms
# import numpy as np
# from tabulate import tabulate
# import yaml
# from PIL import ImageDraw
# from strike_utils import imagenet_classes
# import cv2
# import numpy as np
# import glob
# import re
# from strike_utils import sort_alphanumerically
# import matplotlib.pyplot as plt; plt.rcdefaults()
# import shutil
# import time
# import copy
# import PIL
# #TODO
# #1- ImagePoseData class

# #To delete the .DS_Store file (needed for MacOS ONLY) use the command: find . -name '.#DS_Store' -type f -delete
# jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
# assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508

# def clear_folder(path):
#     if os.path.isdir(path):
#         shutil.rmtree(path)
#         os.mkdir(path)
#     else:
#         os.mkdir(path)

# ############################# ############################# #############################
# ############################# ############################# #############################
# ############################# ############################# #############################
# def rename_img(model_name, outputpath,names_file, names_file5, save_path,pose='pose', correct_class=609):
#     '''
#     save_path = "data/360/ROLL/bg1/airliner_ROLL_360"
#     image_path = "data/360/ROLL/bg1/airliner_ROLL_360/images_lr600"

#     if os.path.isdir(os.path.join(data_root_path[i], 'images_lr600')):
#         data_path = os.path.join(data_root_path[i], 'images_lr600')
#     else:
#         data_path = os.path.join(data_root_path[i], 'images')

#     1-Loads the images from images_path and rename them according to the clasifications (correct/wrong) from the names_file.yml file
#     then save them to the save_path.
#     2-Creates top5 barcharts from the names_file5.yml file.

#     renamed_images_top1
#     renamed_images_top5
#     barplots_top1
#     barplots_top5

#     '''
        
#     if os.path.isdir(os.path.join(save_path, 'images_lr600')):
#         images_path = os.path.join(save_path, 'images_lr600')
#     else:
#         images_path = os.path.join(save_path, 'images')
 

#     k=0
#     with open(names_file5, "r") as ymlfile:
#         result5 = yaml.load(ymlfile)
#     with open(names_file, "r") as ymlfile:
#         result = yaml.load(ymlfile)

#     img_array=[]
#     img_array_top5=[]
#     for image_name  in sort_alphanumerically(list(result.keys())):                                    
#         classification_top5 = result5[image_name][0]
#         classification = result[image_name]

#         pred_class = imagenet_classes[result5[image_name][1][0]].split(',')[0]
#         if k%100==0:
#           print(f'image num {k} ...')
#         k+=1

#         ############# add tags ###############
#         ############# add tags ###############

#         image = Image.open(os.path.join(images_path, image_name)).resize((600,600))
#         image_top5 = copy.deepcopy(image)
#         correct = Image.open('icons/correct.png')
#         correct = correct.resize((70,70))
#         wrong = Image.open('icons/wrong.png')
#         wrong = wrong.resize((70,70))

#         start1 = time.time()

#         mask_im = Image.new("L", correct.size, 0)
#         draw = ImageDraw.Draw(mask_im)
#         draw.ellipse((5, 5, 65, 65), fill=255)
#         if classification == 'correct':
#           image.paste(correct, (280, 10),mask_im)
#         elif classification == 'wrong':
#           image.paste(wrong, (280, 10),mask_im)

#         if classification_top5 == 'correct':
#           image_top5.paste(correct, (280, 10),mask_im)
#         elif classification_top5 == 'wrong':
#           image_top5.paste(wrong, (280, 10),mask_im)
#         # image.save(os.path.join(save_path, 'renamed_images_top1', new_name))

#         ############# new name ###############
#         ############# new name ###############
#         class_name = image_name.split('.')[0].split('_')[0]
#         if pose=='yaw' or pose=='pitch' or pose=='roll':
#             p1 = image_name.split('.')[0].split('_')[-1]
#             p2 = 0
#             new_name = '_'.join([class_name, model_name, pose, p1, pred_class, classification+'.'+image_name.split('.')[1]])
#             new_name_top5 = '_'.join([class_name, model_name, pose, p1, pred_class, classification_top5+'.'+image_name.split('.')[1]])

#         else:
#             p1=image_name.split('.')[0].split('_')[-2]
#             p2=image_name.split('.')[0].split('_')[-1]
#             new_name = '_'.join([class_name, model_name, pose, p1, p2, pred_class, classification+'.'+image_name.split('.')[1]])
#             new_name_top5 = '_'.join([class_name, model_name, pose, p1, p2, pred_class, classification_top5+'.'+image_name.split('.')[1]])

#         ############# bar plot ###############
#         ############# bar plot ###############
#         indices, values = result5[image_name][1], result5[image_name][2] #result5 = {image_name: [correct/wrong , indices, probs]}
#         objects = tuple([imagenet_classes[indices[c]].split(',')[0] for c in range(len(indices))][::-1])
#         color = ['green' if (indices[c] in correct_class) else 'cyan' for c in range(len(indices))]
#         green = False
#         for i, c in enumerate(color):
#           if c=='green' and green==True:
#             color[i] = 'cyan'
#           elif c=='green' and green==False:
#             green = True
#         color = color[::-1]
        
#         probs = values[::-1]
#         y_pos = np.arange(len(objects))
#         plots = plt.barh(y_pos, probs, align='center', alpha=0.5, color=color)

#         plt.yticks(y_pos, objects)
#         plt.xlabel('probability')
#         plt.ylabel('classes')
#         plt.title(f'Top5 probabilities | {pose} degrees: {p1} & {p2}')
#         plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5,alpha=0.8)
#         for bar in plots.patches:
#             plt.text(bar.get_width()+0.01, bar.get_y()+0.4,
#                     str(round((bar.get_width()), 2))+'%',
#                     fontsize=9, fontweight='bold',
#                     color='grey')
#         # plt.savefig(os.path.join(save_path, f'barplots', new_name), bbox_inches='tight', dpi=500)
#         # plt.show()

#         # barplot_img = Image.open(os.path.join(save_path, f'barplots', new_name)).resize((600, 600))
#         plt.tight_layout()
#         fig = plt.gcf()
#         fig.canvas.draw()
#         barplot_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).resize((600, 600))
#         plt.close()


#         img2 = Image.new("RGB", (600*2, 600), "white")
#         img2.paste(image, (0, 0))  
#         img2.paste(barplot_img, (600, 0))
#         # img2.save(os.path.join(save_path, f'top1_joined_renamed_images', new_name))
#         opencvImage = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
#         img_array.append(opencvImage)
      
#         img2 = Image.new("RGB", (600*2, 600), "white")
#         img2.paste(image_top5, (0, 0))  
#         img2.paste(barplot_img, (600, 0))
#         # img2.save(os.path.join(save_path, f'top5_joined_renamed_images', new_name_top5))      

#         opencvImage = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
#         img_array_top5.append(opencvImage)
   
    
#     print('saving  the video')
#     size=(600,600)
#     out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_video_top1.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

#     for i in range(len(img_array)):
#         out.write(img_array[i])
#     out.release()

#     out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_fast_video_top1.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
#     for i in range(len(img_array)):
#         out.write(img_array[i])
#     out.release()

#     ##top5 videos

#     print('saving  the video')

#     out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_video_top5.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
#     print(img_array)
#     for i in range(len(img_array_top5)):
#         out.write(img_array_top5[i])
#     out.release()

#     out = cv2.VideoWriter(os.path.join(outputpath, model_name, model_name+f'_fast_video_top5.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
#     for i in range(len(img_array_top5)):
#         out.write(img_array_top5[i])
#     out.release()

# if __name__=='__main__':
#     device = "cuda" if torch.cuda.is_available() else "cpu"
 
#     # models_names = {'bit':'BiTM-resnetv2-152x2', 'swsl':'ResNeXt101-32x16d-swsl', 'simclr':'simclr-resnet50x1', 'vit':'vit-large-patch16-224'}
#     # model_names = ['BiTM-resnetv2-152x2', 'Clip-ViT-B-32','resnet50', 'resnet152','ResNeXt101-32x16d-swsl', 'simclr-resnet50x1', 'vit-large-patch16-224']

# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#     model_names = ['BiTM-resnetv2-152x2']
#     jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
#     assault_rifle=413; whiteshark=2; cannon=471; mug=504; keyboard=508; tablelamp=846; lampshade=619;

#     correct_class = [[airliner],[airliner],[airliner]]
#     obj='airliner'
#     pose = ['pitch','pitch','pitch']

#     savepath  = [f"data/360/PITCH result/result/bg1/{obj}_PITCH_360/", f"data/360/PITCH result/result/bg2/{obj}_PITCH_360/", f"data/360/PITCH result/result/nobg/{obj}_PITCH_360/"]
#     data_root_path = [f"data/360/PITCH/bg1/{obj}_PITCH_360", f"data/360/PITCH/bg2/{obj}_PITCH_360", f"data/360/PITCH/nobg/{obj}_PITCH_360"]
    
#         ######################################################################
#     for m in range(len(model_names)):
#         print('current model : ',model_names[m])
#         for i in range(len(savepath)):
#           start = time.time()

#           print(f'working on object number {i+1}/{len(savepath)}')

#           # rename the images with the classifications 

#           rename_img(model_name=model_names[m], outputpath=savepath[i],
#                     names_file=os.path.join(savepath[i], model_names[m],model_names[m]+'_result.yml'), names_file5=os.path.join(savepath[i],model_names[m], model_names[m]+'_result5.yml'), 
#                     save_path=data_root_path[i], pose=pose[i], correct_class=correct_class[i])


#           print('total time per bg: ',time.time()-start)