from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate

jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879;
TRUE_CLASS_list = [jeep, ambulance, forklift, jeep]
# objpath = "objects/jeep/jeep.obj"
# mtlpath = "objects/jeep/jeep.mtl"
savepath_list = ["data/360/ROLLPITCH/bg1/jeep_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/ambulance_ROLLPITCH_360/images",
                 "data/360/ROLLPITCH/bg1/loader_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/brownjeep_ROLLPITCH_360/images"]
objectpath_list = [("objects/jeep/jeep.obj", "objects/jeep/jeep.mtl"), ("objects/ambulance/ambulance.obj", "objects/ambulance/ambulance.mtl"),
                        ("objects/loader/loader.obj", "objects/loader/loader.mtl"), ("objects/brownjeep/brownjeep.obj", "objects/brownjeep/brownjeep.mtl")]

# savepath='data/inceptionv3 default 299 -6'
bgpath_list = ["backgrounds/medium.jpg", "backgrounds/medium.jpg", "backgrounds/medium.jpg", "backgrounds/medium.jpg", "backgrounds/medium.jpg"]
pose = 'rollpitch'

if __name__ == "__main__":
    # Initialize neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on ',device)
    model = Model(device,model_name='inceptionv3').to(device) 
    for obj in range(len(savepath_list)):
      objpath, mtlpath = objectpath_list[obj]
      savepath = savepath_list[obj]
      bgpath = bgpath_list[obj]
      TRUE_CLASS = TRUE_CLASS_list[obj]
      print(f'working on object: {obj} ...........')
      # Initialize renderer.
      renderer = Renderer(
          objpath, mtlpath, bgpath
      )
      # print(f'Default renderer parameters: ',renderer.prog["x"].value)
      # Render scene.
      save = True
      
      
      #########################    #########################
      #########################    #########################
      renderer.prog["x"].value = 0
      renderer.prog["y"].value = 0
      renderer.prog["z"].value = -6
      renderer.prog["amb_int"].value = 0.3
      renderer.prog["dif_int"].value = 0.9
      DirLight = np.array([1.0, 1.0, 1.0])
      DirLight /= np.linalg.norm(DirLight)
      renderer.prog["DirLight"].value = tuple(DirLight)
      correct = 0
      correct5 = 0
      correct_t_conf=[]
      wrong_conf=[]
      correct_f_conf=[]

      log = open (os.path.join(savepath,"log.txt"), "a")

      for i in range(0,360,10):
          degreei = i * (np.pi / 180)
          for j in range(0,360,10):
            # Alter renderer parameters.
            # degreej = j * (np.pi / 180)
            R_obj = gen_rotation_matrix(degreei, degreej, 0)  #yaw, pitch, roll
            #  R_obj =
            #  [[ 0.70710678  0.5         0.5       ]
            #  [ 0.          0.70710678 -0.70710678]
            #  [-0.70710678  0.5         0.5       ]]
            # print('R_obj ',R_obj)
            renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
            # Render new scene.
            image = renderer.render()
            # Get neural network probabilities.
            with torch.no_grad():
                out = model(image)

            probs = torch.nn.functional.softmax(out, dim=1)
            print('inceptionv3 probability for the correct class',probs[0][TRUE_CLASS].item())
            values5, indices5 = probs[0].topk(5)
            if TRUE_CLASS in indices5:
              correct5+=1

            probs_np = probs[0].detach().cpu().numpy()
            max_index = probs_np.argmax()
            if max_index == TRUE_CLASS:
                correct+=1
                correct_t_conf.append(probs[0][TRUE_CLASS].item())
                print(f'{i} {j}: correct')
                if save:
                  image.save(os.path.join(savepath,f'pose_{i}_correct.png'))
                else:
                  image.show()
            else:
                wrong_conf.append(probs[0][max_index].item())
                correct_f_conf.append(probs[0][TRUE_CLASS].item())

                print(f'{i} {j}: wrong')
                if save:
                  image.save(os.path.join(savepath,f'pose_{i}_wrong.png'))
                else:
                  image.show()
              

          print(f'pose: {i}-{j}, accuracy : {correct}/360, top5: {correct5}, correct_t_conf: {np.mean(correct_t_conf)}, wrong_conf: {np.mean(wrong_conf)}, correct_f_conf: {np.mean(correct_f_conf)}')
          log.write(f'pose: {i}-{j}, accuracy : {correct}/360, top5: {correct5}, correct_t_conf: {np.mean(correct_t_conf)}, wrong_conf: {np.mean(wrong_conf)}, correct_f_conf: {np.mean(correct_f_conf)} \n')

      correct_t_conf = np.mean(correct_t_conf)
      wrong_conf = np.mean(wrong_conf)
      correct_f_conf = np.mean(correct_f_conf)
      num_images = 360
      data =[[pose, num_images, correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, model.model_name]] 
      table = tabulate(data, headers=['pose', 'num_images', 'top1', 'top5', "correct_t_conf", "correct_f_conf", "wrong_conf", 'model'])

      # with open (os.path.join(savepath,"log.txt"), "w") as log:
      log.write('\n'+table+'\n')
      log.close()
      print (table)

