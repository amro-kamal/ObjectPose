from renderer import Renderer
from strike_utils import *
import os
import numpy as np
import shutil


# 1-choose an object
# 2-choose a pose
# 3-generate the images for all the bgs

TRUE_CLASS_list = ['barberchair']
bgs =  ['bg2', 'bg1', 'nobg']
num_images = 180
scale_range = [-34, -11]
amb_int = 0.8

for bg in bgs:
    print('working with ', bg)
        
    savepath_list = [f"newdata/360/scale_and_rot/{TRUE_CLASS_list[0]}_360/{bg}/images"]
    objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
    bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg"] if os.path.exists(f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg") else [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpeg"] 

    if os.path.exists(f"newdata/360/scale_and_rot/{TRUE_CLASS_list[0]}_360/{bg}/images"):
        shutil.rmtree(f"newdata/360/scale_and_rot/{TRUE_CLASS_list[0]}_360/{bg}/images")

    os.makedirs(f"newdata/360/scale_and_rot/{TRUE_CLASS_list[0]}_360/{bg}/images", exist_ok=True)

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
            renderer.prog["x"].value = 0.0
            renderer.prog["y"].value = 0.0

            
            renderer.prog["amb_int"].value = amb_int#light
            renderer.prog["dif_int"].value = 1.0
            DirLight = np.array([1.0, 1.0, 1.0])
            DirLight /= np.linalg.norm(DirLight)

            renderer.prog["DirLight"].value = tuple(DirLight)
                
            for i in range(num_images):
                z_scale = np.random.randint(scale_range[0], scale_range[1]+1) 
                renderer.prog["z"].value = z_scale
                degreey = np.random.randint(0,360) * (np.pi / 180) 
                degreer = np.random.randint(0,360) * (np.pi / 180) 
                degreep = np.random.randint(0,360) * (np.pi / 180) 


                R_obj = gen_rotation_matrix(degreey, degreep, degreer) #yaw, pitch, roll # ROLLphi @ YAWThetaMat 

                renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
                # Render new scene.
                image = renderer.render()

                image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{bg}_{int(degreey*(180/np.pi))}_{int(degreer*(180/np.pi))}_{int(degreep*(180/np.pi))}_{abs(z_scale)}.png'))

            print('images saved to ',savepath_list[obj])

            del renderer