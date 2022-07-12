import shutil
from renderer import Renderer
from strike_utils import *
import os
from tabulate import tabulate
import numpy as np
jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;


# 1-choose an object
# 2-choose a pose
# 3-generate the images for all the bgs

TRUE_CLASS_list = ['hammerhead']
POSE = 'scaling'
init = '2'
bgs = ['bg1', 'bg2', 'nobg']
z_range = list(range(-34, -9, 1))
amb_int = 0.8

for bg in bgs:
    pose = bg+'_scaling' 

    obj=TRUE_CLASS_list[0]

    savepath_list = [f"newdata/360/scaling/{TRUE_CLASS_list[0]}_{POSE}_360/{bg}/images"]
    objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
    bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg"] if os.path.exists(f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg") else [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpeg"] 

    if __name__ == "__main__":

        # Initialize neural network.
        for obj in range(len(savepath_list)):
            objpath, mtlpath = objectpath_list[obj]
            savepath = savepath_list[obj]
            bgpath = bgpath_list[obj]
            TRUE_CLASS = TRUE_CLASS_list[obj]
            os.makedirs(savepath, exist_ok=True)

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
            for z in z_range:
                renderer.prog["x"].value = 0
                renderer.prog["y"].value = 0
                renderer.prog["z"].value =  z
                
                renderer.prog["amb_int"].value = amb_int #light
                renderer.prog["dif_int"].value = 1.0
                DirLight = np.array([1.0, 1.0, 1.0])
                DirLight /= np.linalg.norm(DirLight)

                renderer.prog["DirLight"].value = tuple(DirLight)


                R_obj = gen_rotation_matrix(-np.pi/8, np.pi/10, 0) #yaw, pitch, roll

                renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
                # Render new scene.
                image = renderer.render()

                image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose+init}_{abs(z)}.png'))

            print('images saved to ',savepath_list[obj])

        del renderer