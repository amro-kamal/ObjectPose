from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate
import numpy as np
jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;


# 1-choose an object
# 2-choose a pose
# 3-generate the images for all the bgs

TRUE_CLASS_list = ['fireengine']
POSE = 'ROLLPITCH' #'IN_PLANE_ROLL'
pose = 'rollpitch' #in_plane_roll' 
bgs = ['bg1', 'bg2', 'nobg']
amb_int  = 0.75
x, y, z= 0, -0.1, -8.5
for bg in bgs:
    
    savepath_list = [f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"]
    objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
    bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg"] if os.path.exists(f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg") else [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpeg"] 

    os.makedirs(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images", exist_ok=True)

    if __name__ == "__main__":
        
        # Initialize neural network.

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
            
            #########################    #########################
            #########################    #########################
            renderer.prog["x"].value = x
            renderer.prog["y"].value = y
            renderer.prog["z"].value = z
            
            renderer.prog["amb_int"].value = amb_int#light
            renderer.prog["dif_int"].value = 1.0
            DirLight = np.array([1.0, 1.0, 1.0])
            DirLight /= np.linalg.norm(DirLight)

            renderer.prog["DirLight"].value = tuple(DirLight)
                
            for i in range(0,360,36):
                degreey = i * (np.pi / 180) 
                for j in range(0,360,36):
                    # Alter renderer parameters.
                    degreep = j * (np.pi / 180)


                    R_obj = gen_rotation_matrix(np.pi/10, degreey+np.pi/20, degreep+np.pi/40) #yaw, pitch, roll # ROLLphi @ YAWThetaMat 

                    renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
                    # Render new scene.
                    image = renderer.render()

                    image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{i}_{j}.png'))

            print('images saved to ',savepath_list[obj])


    del renderer