from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate
import numpy as np
from generate_rot_bg360 import rot_image_ramdomly

jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;

# 1-choose an object
# 2-choose a pose
# 3-generate the images for all the bgs


TRUE_CLASS_list = ['hammerhead'] 
POSE = 'ROLL' #IN_PLANE_ROLL
pose = 'roll' #in_plane_roll
bgs =  ['bg1', 'bg2', 'nobg'] 
amb_int  = 0.75
x, y, z= 0, -0.1, -8.5

for bg in bgs:
    print(bg)

    YAWTheta = np.pi/1.5
    ROLLTheta = -np.pi/15
    PITCHTheta = -np.pi/15

    savepath_list = [f"newdata/360/rot_rand_roll/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"] #rot_rand_in_plane_roll #rot_rand_roll
    objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
    bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg"] if os.path.exists(f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg") else [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpeg"] 

    os.makedirs(f"newdata/360/rot_rand_roll/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images", exist_ok=True)

    if __name__ == "__main__":

        for obj in range(len(savepath_list)):
            objpath, mtlpath = objectpath_list[obj]
            savepath = savepath_list[obj]
            bgpath = bgpath_list[obj]
            TRUE_CLASS = TRUE_CLASS_list[obj]
            print(f'working on object: {objectpath_list[obj][0]} ...........')
            # Initialize renderer.
            if bg=='nobg':
                renderer = Renderer(objpath, mtlpath)
            else:
                renderer = Renderer(
                objpath, mtlpath, bgpath
                )
            
            # print(f'Default renderer parameters: ',renderer.prog["x"].value)
            #########################    #########################
            #########################    #########################
            renderer.prog["x"].value = x
            renderer.prog["y"].value = y
            renderer.prog["z"].value = z

            renderer.prog["amb_int"].value = amb_int #light
            renderer.prog["dif_int"].value = 1.0
            DirLight = np.array([.3, 1.0, 1.0])
            DirLight /= np.linalg.norm(DirLight)

            renderer.prog["DirLight"].value = tuple(DirLight)
            
            YAWThetaMat = np.array( [[ np.cos(YAWTheta) , 0.    ,       np.sin(YAWTheta) ],
                                    [ 0.               , 1.    ,       0.               ],
                                    [-np.sin(YAWTheta) , 0.    ,       np.cos(YAWTheta) ]])
                                           
            YAW_90 = np.array(       [[ np.cos(-90 * (np.pi / 180) ) , 0.    ,       np.sin(-90 * (np.pi / 180) )   ],
                                    [ 0.               , 1.    ,       0.               ],
                                    [-np.sin(-90 * (np.pi / 180) ) , 0.    ,       np.cos(-90 * (np.pi / 180) )   ]])

            ROLLThetaMat = np.array( [[ np.cos(ROLLTheta) ,-np.sin(ROLLTheta)       , 0  ],
                                        [ np.sin(ROLLTheta) , np.cos(ROLLTheta)       , 0  ],
                                        [ 0.                , 0.                      , 1. ]])
        
            PITCHThetaMat = np.array( [[  1,                  0     ,                0     ],
                                        [  0,     np.cos(PITCHTheta) , -np.sin(PITCHTheta)  ],
                                        [  0,     np.sin(PITCHTheta) ,  np.cos(PITCHTheta)  ]])
            
            for i in range(0,360,360):
                degreey = i * (np.pi / 180) 
                for j in range(0,360,2):
                    # Alter renderer parameters.
                    degreep=0
                    bg_degreep=0



                    while abs(degreep-bg_degreep)<45 or abs(degreep-bg_degreep)>315 : 
                        degreep =  np.random.randint(10,351)      #rand angle for the object
                        bg_degreep = np.random.randint(10,351)    #rand angle for the bg
                    # print(degreep, bg_degreep)
                    degreep_name = degreep
                    degreep =  degreep * (np.pi / 180)  
                    # degreep =  degreep+np.pi/15

                    random_name = int(np.random.randint(360))
                    if bg != 'nobg':    
                        bgImage  = Image.open(bgpath) #get the bg
                        rot_image_ramdomly(bgImage, bg_degreep, savepath=savepath, savename=f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{random_name}_{int(bg_degreep)}_{int(degreep_name)}.png')#rot the bg
                        renderer.background_f = os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{random_name}_{int(bg_degreep)}_{int(degreep_name)}.png')
                        renderer.set_up_background(renderer.background_f)
                    
                    ROLLphi = np.array( [[ np.cos(degreep) ,-np.sin(degreep)         , 0  ],
                                        [ np.sin(degreep) , np.cos(degreep)         , 0  ],
                                        [ 0.              , 0.                      , 1. ]])

                    PITCHphi = np.array([[ 1,                 0    ,         0         ],
                                        [  0,     np.cos(degreep) , -np.sin(degreep)  ],
                                        [  0,     np.sin(degreep) ,  np.cos(degreep)  ]])

                    R_obj = gen_rotation_matrix(-np.pi/10, -np.pi/10, degreep) #yaw, pitch, roll

                    renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
                    # Render new scene.
                    image = renderer.render()
                    # image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{int(bg_degreep)}.png'))
                    image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{random_name}_{int(bg_degreep)}_{int(degreep_name)}.png'))

            print('images saved to ', savepath_list[obj])

    del renderer