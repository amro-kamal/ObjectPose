import os
from PIL import Image, ImageDraw
import numpy as np
def generate_rot_data(colorImage, pose, true_class, savepath):
  for degree in range(0, 360, 2):
    height, width, _  = np.array(colorImage).shape
    rotated_img_arr = np.array(colorImage.rotate(degree, resample=Image.BICUBIC ))
    black_cirlce = Image.new('RGB', [height,width] , color=(123, 116, 103)) #black image
    draw = ImageDraw.Draw(black_cirlce) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 0, outline = "black")

    white_circle = Image.new('L', [height,width] , 0) #black image
    draw = ImageDraw.Draw(white_circle) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 255, outline = "white")
    norm_white_circle = (np.array(white_circle).reshape(height,width,1)/255).astype(np.uint8)

    img_circle = rotated_img_arr*norm_white_circle

    final_img = Image.fromarray(np.array(black_cirlce)+img_circle)

    final_img.save(os.path.join(savepath, f'{true_class}_{pose}_{degree}.png'), quality=100, subsampling=0)


#change the TRUE_CLASS and pose only
TRUE_CLASSs = [ 'airliner' ,     'fireengine'  ,  'garbagetruck' ,  'mountainbike'  , 'shoppingcart'  , 'tractor',
'barberchair'  , 'foldingchair' , 'hammerhead'  ,  'parkbench'    , 'tablelamp'    ,  'wheelbarrow',
'cannon'   ,  'forklift' ,     'jeep'  ,     'rockingchair' ,  'tank' ]

bg='nobg'

for TRUE_CLASS in TRUE_CLASSs:
  pose='roll'
  print(TRUE_CLASS)
  ####################################
  # savepath = f"data/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/rot_both_images"
  # colorImage  = Image.open(f"data/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/baseline_images/{TRUE_CLASS}_{pose}_0_0.png")

  # generate_rot_data(colorImage, pose=pose, true_class=TRUE_CLASS, savepath=savepath)

  # savepath = f"data/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/rot_both_images_lr600"
  # colorImage  = Image.open(f"data/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/baseline_images_lr600/{TRUE_CLASS}_{pose}_0_0.png")

  # generate_rot_data(colorImage, pose=pose, true_class=TRUE_CLASS, savepath=savepath)
  if not os.path.exists(f"newdata/rot_both360/ROLL/{bg}"):
    os.mkdir(f"newdata/rot_both360/ROLL/{bg}")

  if not os.path.exists(f"newdata/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360"):
    os.mkdir(f"newdata/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360")
    os.mkdir(f"newdata/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/rot_both_images")

  savepath = f"newdata/rot_both360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/rot_both_images"
  colorImage  = Image.open(f"newdata/360/ROLL/{bg}/{TRUE_CLASS}_ROLL_360/images/{TRUE_CLASS}_{pose}_{bg}_0.png")

  generate_rot_data(colorImage, pose=pose, true_class=TRUE_CLASS, savepath=savepath)

