import os
from PIL import Image, ImageDraw
from matplotlib.pyplot import savefig
import numpy as np
from strike_utils import true_class_dict
from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate

def rot_image(coloredImage, degree, savepath, savename):
    '''
    function to rotate an image by <degree> degrees

    '''
    height, width, _  = np.array(coloredImage).shape
    rotated_img_arr = np.array(coloredImage.rotate(degree, resample=Image.BICUBIC ))
    black_cirlce = Image.new('RGB', [height,width] , color=(123, 116, 103)) #black image
    draw = ImageDraw.Draw(black_cirlce) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 0, outline = "black")

    white_circle = Image.new('L', [height,width] , 0) #black image
    draw = ImageDraw.Draw(white_circle) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 255, outline = "white")
    norm_white_circle = (np.array(white_circle).reshape(height,width,1)/255).astype(np.uint8)

    img_circle = rotated_img_arr*norm_white_circle

    final_img = Image.fromarray(np.array(black_cirlce)+img_circle)

    final_img.save(os.path.join(savepath, f'{savename}'), quality=100, subsampling=0)

def generate_rot_data(coloredImage, pose, true_class, savepath):
  paths = []
  for degree in range(0, 360, 2):
    rot_image(coloredImage, degree, savepath, f'{true_class}_{pose}_{degree}.png')
    paths.append(os.path.join(savepath, f'{true_class}_{pose}_{degree}.png'))
  
  return paths



#change the obj and pose only
init = '2'
bgs = ['bg1', 'bg2']
POSE = 'bg_rot'
amb_int  = 1.2
x, y, z= 0, -0.1, -8.5

for obj in ['parkbench']: #true_class_dict.keys():

  objectpath_list = (f"objects/{obj}/{obj}.obj", f"objects/{obj}/{obj}.mtl")
  # Initialize neural network.
  objpath, mtlpath = objectpath_list
  print(f'working on object: {objpath} ...........')
  obj = obj.lower()
  renderer = Renderer(objpath, mtlpath)

  for bg in bgs:
      pose= bg+'_rot' #bg1_rot
      ####################################
      savepath = f"newdata/360/bg_rot/{obj}_{POSE}_360/{bg}/images"
      os.makedirs(savepath, exist_ok=True)

      upritght_bg_path  = f"backgrounds/{obj}_{bg}_500.jpg" if os.path.exists(f"backgrounds/{obj}_{bg}_500.jpg") else f"backgrounds/{obj}_{bg}_500.jpeg"
      bgImage  = Image.open(upritght_bg_path)

      #360 rotated background 
      rotated_180_bgs_path = generate_rot_data(bgImage, pose=pose+'_'+init, true_class=obj, savepath=savepath)
      print('backgrounds generated')

      for i, bgpath in enumerate(rotated_180_bgs_path):
          if i%100==0: print(f'{i} images generated')

          renderer.set_up_background( background_f= bgpath)

          renderer.prog["x"].value = x
          renderer.prog["y"].value = y
          renderer.prog["z"].value = z
          
          renderer.prog["amb_int"].value = amb_int #light
          renderer.prog["dif_int"].value = 1.0
          DirLight = np.array([1., 1.0, 1.0])
          DirLight /= np.linalg.norm(DirLight)
          renderer.prog["DirLight"].value = tuple(DirLight)
          #########################    #########################

          # Alter renderer parameters.
          R_obj = gen_rotation_matrix( -np.pi/5, 0, -np.pi/15)  #yaw, pitch, roll

          renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
          # Render new scene.
          image = renderer.render()

          image.save(bgpath)
      print('images saved to ',bgpath)
      # del renderer


print("DONE")











