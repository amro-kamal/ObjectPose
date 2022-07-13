### Progress and limitations of deep networks to recognize objects in unusual poses
by Amro Abbas & Stephane Deny  
ArXiv Link:  

# ObjectPose Dataset
ُThe Datasets folder contains the code for creating ObjectPose datasets. The code for rendering 3d objects in this repo is from [Strike (With) A Pose](https://github.com/airalcorn2/strike-with-a-pose) repo.

**Some Samples from the data:**

<p float="left">
  <img src="Datasets/README images/airliner_roll_bg1_14.png" width="150" />
  <img src="Datasets/README images/cannon_roll_bg1_116.png" width="150" /> 
  <img src="Datasets/README images/tank_yaw_nobg_86.png" width="150" />
  <img src="Datasets/README images/tractor_yaw_nobg_132.png" width="150" />
  <img src="Datasets/README images/jeep_pitch_bg1_108.png" width="150" />

</p>
<p float="left">
  <img src="Datasets/README images/barberchair_roll_bg1_2.png" width="150" />
  <img src="Datasets/README images/fireengine_roll_bg1_6.png" width="150" /> 
  <img src="Datasets/README images/shoppingcart_yaw_nobg_14.png" width="150" />
  <img src="Datasets/README images/tablelamp_yaw_nobg_80.png" width="150" />
  <img src="Datasets/README images/rockingchair_yaw_nobg_24.png" width="150" /> 
</p><p float="left">
  <img src="Datasets/README images/parkbench_pitch_bg1_192.png" width="150" />
  <img src="Datasets/README images/mountainbike_pitch_bg1_64.png" width="150" /> 
  <img src="Datasets/README images/hammerhead_pitch_bg1_28.png" width="150" />
  <img src="Datasets/README images/forklift_roll_bg1_0.png" width="150" />
  <img src="Datasets/README images/barberchair_roll_bg1_2.png" width="150" />

</p><p float="left">
    <img src="Datasets/README images/foldingchair_pitch_bg1_40.png" width="150" />
    <img src="Datasets/README images/wheelbarrow_yaw_nobg_42.png" width="150" />
</p>
In addition to the main ObjectPose dataset, the paper introduced a set of similar datasets to test to robustness to different tansformations including scaling, rotation, background rotation, etc. See the paper for more details about the datasets.

</p><p float="left">
    <img src="Datasets/README images/objectpose truck samples.jpg" width="400" />
    <img src="Datasets/README images/scaling + three-axes + compined + bg rot samples.jpg" width="400" />
</p>


# Deep Networks

## How to run the models on ImageNetV2 dataset:
1. Create (only once) and activate python environment (conda create -n "env name") then (conda activate "env name"). Alternatively, if you are not using conda, you can use venv to create the env [link](https://oit.utk.edu/hpsc/isaac-open/pip-and-venv/).
2. Clone the repo (git clone https://github.com/amro-kamal/ObjectPose.git). This will clone a folder "ObjectPose".
4. cd to ObjectPose directory.
3. Install the requirements (pip install -r requirements.txt).
4. Clone CLIP repo to the ```ObjectPose/models``` folder:

```
cd ObjectPose/src/models                  #go to the models folder 
git clone https://github.com/openai/CLIP  #clone CLIP repo
cd ..                                     #go back to the src folder
```

5. Download ImageNetV2 dataset [from here](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz) (1.2GB). Create a folder (call it data for example) and add the dataset folder there (Remenber this dataset path because we will pass it to the code later). 
6. Make sure you are at the ```ObjectPose/src``` folder (cd src). Then run one of the commands in the commands.txt files based on the dataset you want to use. For ```python run_imagenet_v2.py``` for ImageNetv2 dataset. This command explained under the **"Testing on ImageNetV2"** section below.


## Install the requirements:
```
pip install -r requirements.txt

```
The command will install all the requirements for you. See the requirements.txt for all the rquirements. The code works fine with Python 3.7.13.

## Testing on ImageNetV2
To test the models on ImageNetV2 dataset, go to src folder and run the command:

```
python run_imagenet_v2.py --batchsize=16
                          --dataroot='../../data/imagenetv2-matched-frequency/data' 
                          --allresultpath = '../../data/imagenetv2-matched-frequency/all_results'
                          --datasetname='imagenetv2' 
                          --modelslist = 'all'
                          --testcode
                          --crop
                          

```
**--dataroot** => Dataset path. Change it to your custom path.

**--allresultpath** => Path to save the result. The code will save a .txt file containing the top1 and top5 accuracy. Change it to your custom path.

**--datasetname**='imagenetv2' for ImageNetV2 | 'co3d' for CO3D.

**--testcode** => When this flag is set to True the code will run for one batch only, just to test the code. After you test your code make sure to set it to False.

**--modelslist** => If 'all', all the models in the model.py file will be tested (currently 37 models). If you want to test specific models, pass a string containing the models' names (space sperated), for example    --modelslist = 'ViT_21k_L16 ViT_21k_B16'. See the [model.py script](https://github.com/amro-kamal/ObjectPose/blob/main/src/models/models.py) for the all the available models.


## Testing on ObjectPose:
To test the models on ObjectPose dataset, go the src folder and run the command:

```
python run_objectpose.py --batchsize=16 --dataroot="../../data/ObjectPose_dataset/ObjectPose" --saveroot="../../data/ObjectPose_dataset/Experiments Results/ObjectPose" --poses="ROLL YAW PITCH" --modelslist='all' --bgs="bg1 bg2 nobg" --crop

```
**--poses** => Rotation axes used for creating the data. See commands.txt file for poses corresponding to each dataset.

**--bgs** => Background images: this defines which part of the data to use. Note that not all the datasets use three background images. See commands.txt file.


<!-- ## Models Table: List of models tested in the paper

| Model         | Source | Dataset | Params | IN acc |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Resnet50  | torchvision | ImageNet(1M) | 25M | 76.1% 
| Resnet152 | torchvision | ImageNet(1M) | 69M | 77.3% 
| Resnet101 | torchvision  | ImageNet(1M) | 43M | 78.3% 
| | | | | | |
| CLIP-ViT-B16 | CLIP |  WT(400M) | 86M | 63.2% 
| CLIP-RN50     | CLIP |  WT(400M) | 25M  | 62.2% 
| CLIP-RN101     | CLIP |  WT(400M) | 43M | 59.6% 
| | | | | | |
| ViT-L16 | timm |  ImageNet(1M) | 307M | 77% 
| ViT-B16 | timm |  ImageNet(1M) | 86M  | 78% 
| ViT-S16 | timm |  ImageNet(1M) | 22M  | 75%  
| ViT-B16-sam | timm |  ImageNet(1M) | 86M | 79.9% 
| ViT-21k-B16 | pretrained-vit | ImageNet21k(14M) | 86M | 84% 
| ViT-21k-L16 | pretrained-vit | ImageNet21k(14M) | 307M | 85% 
| | | | | | |
| SWIN-B-384 | timm | ImageNet21k(14M)  | 88M  | 86.4% 
| SWIN-L     | timm | ImageNet21k(14M) | 197M | 86.3% 
| SWIN-L-384 | timm | ImageNet21k(14M)  | 197M | 87.3% 
| | | | | | |
| Simclr | timm | ImageNet(1M) | 25M | 68.9% 
| | | | | | |
| BiTM-RN50  | timm  | ImageNet21k(14M) | 25M | 80.0% 
| BiTM-RN101 | timm  | ImageNet21k(14M) | 43M | 82.5% 
| BiTM-RN152x2 | timm  | ImageNet21k(14M) | 98M | 85.5% 
| | | | | | |
| SWSL-ResNet50  | torchhub  | (64M) | 25M | 79.1% 
| SWSL-ResNeXt101 | torchhub  | (64M) | 193M | 81.2% 
| | | | | | |
| Mixer-B16 | timm | ImageNet(1M) | 59M | 76.44% 
| Mixer-L16 | timm |  ImageNet(1M) | 207M | 71.76% 
| | | | | | |
| BEiT-B16 | transformers | ImageNet21k(14M) | 87M | 85.2%  
| BEiT-L16 | transformers |  ImageNet21k(14M) | 304M | 87.4% 
| | | | | | |
| Deit-B16 | timm |  ImageNet(1M) | 86M | 83.4% 
| Deit-S16 | timm |  ImageNet(1M) | 22M | 81.2%  
| | | | | | |
| EffN-B7-NS | torchhub | JFT(300M) | 66M | 86.9% 
| EffN-L2-NS | torchhub |   JFT(300M) | 480M|  88.4% 
| | | | | | | 
| ConvNeXt-XL | timm |  ImageNet21k(14M) | 350M | 87.0% 
| ConvNeXt-L-384 | timm | ImageNet21k(14M) | 198M | 87.5%  
| ConvNeXt-L | timm |  ImageNet21k(14M) | 198M | 86.6% 
| ConvNeXt-L | timm |  ImageNet21k(1M) | 306M | 82.6% 
| | | | | | |
| ConVit-B | timm | ImageNet(1M) | 86M | 82.4% 
| ConVit-S | timm | ImageNet(1M) | 27M | 81.3% 
| | | | | | |
| SWAG-ViT-L16-512 | torchhub | IG(3.6B) | 305M | 88.07% 
| SWAG-ViT-H14-512 | torchhub | IG(3.6B) | 633M | 88.55% 
| SWAG-RegNetY-128GF-384 | torchhub | IG(3.6B) | 645M | 88.23%
| | | | | | |

 -->
