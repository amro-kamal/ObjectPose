The code for rendering 3d objects in this repo is from the [Strike (With) A Pose](https://github.com/airalcorn2/strike-with-a-pose) repo.
# ObjectPose Dataset
Are vision models robust against Unusual poses?


<!-- | Model         | Source | Model Name| Dataset | Params | IN acc | repo |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Resnet50  | torchvision | ResNet50 | ImageNet(1M) | 25M | 79.3% |
| Resnet152 | torchvision | ResNet152 | ImageNet(1M) | 69M | 80.1% |
| Resnet101 | torchvision | ResNet101 | ImageNet(1M) | 43M | 80.1% |
| | | | | | |
| Clip_vit_B16 | clip | CLIP_ViT_B_16 | WebImageText(400M) | 86M | 63.2% |
| Clip_50      | clip | CLIP_ResNet50 | WebImageText(400M) | 25M  | 62.2% |
| Clip_101     | clip | CLIP_ResNet101 | WebImageText(400M) | 43M | 59.6% |
| | | | | | 
| ViT_L16 | timm | ViT_L_patch16_224 | ImageNet(1M) | 307M | 77% |
| ViT_B16 | timm | ViT_B_patch16_224 | ImageNet(1M) | 86M  | 78% |
| ViT_S16 | timm | ViT_S_patch16_224 | ImageNet(1M) | 22M  | 75%  |
| ViT_B16_sam | timm | ViT_B_patch16_sam_224 | ImageNet(1M) | 86M | 79.9% |
| | | | | | |
| ViT_21k_B16 | pytorch_pretrained_vit | ViT_21k_base16_384 | ImageNet21k(14M) | 86M | 84% | https://github.com/lukemelas/PyTorch-Pretrained-ViT
| ViT_21k_L16 | pytorch_pretrained_vit | ViT_21k_large16_384 | ImageNet21k(14M) | 307M | 85% | https://github.com/lukemelas/PyTorch-Pretrained-ViT
| | | | | | |
| SWIN_B     | timm | swin_base_patch4_window7_224 | ImageNet21k(14M) | 88M  | 85.2% | 
| SWIN_B_384 | timm | swin_base_patch4_window12_384 | ImageNet21k(14M) | 88M  | 86.4% |
| SWIN_L     | timm | swin_large_patch4_window7_224 | ImageNet21k(14M) | 197M | 86.3% |
| SWIN_L_384 | timm | swin_large_patch4_window12_384 | ImageNet21k(14M) | 197M | 87.3% |
| | | | | | |
| DINO_S16 | torchhub | DINO_ViT_small_16 | xx | xx | xx
| DINO_B16 | torchhub | DINO_ViT_base_16 | xx | xx | xx
| DINO_RN50 | torchhub | DINO_ResNet50 | xx | xx | xx
| | | | | | |
| Simclr | timm | SimCLR_ResNet50 | ImageNet(1M) | 25M | 69.3% |
| Moco | timm | MOCO_ResNet50 | ImageNet(1M) | 25M | 71.1% | 
| | | | | | |
| BiTM_50  | timm | BiTM_resnetv2_50x1 | ImageNet21k(14M) | 25M | 80.0% |
| BiTM_101 | timm | BiTM_resnetv2_101x1 | ImageNet21k(14M) | 43M | 82.5% |
| BiTM_152x2 | timm | BiTM_resnetv2_152x2 | ImageNet21k(14M) | 98M | 85.5% |
| | | | | | |
| SWSL_ResNet50  | timm | ResNet50_swsl | (64M) | 25M | 79.1% |
| SWSL_ResNeXt101 | timm | ResNeXt101_32x16d_swsl | (64M) | 193M | 81.2% |
| | | | | | |
| Mixer_S16 | timm | Mixer_small_16_224 | ImageNet(1M) | 18M | xx |
| Mixer_B16 | timm | Mixer_base_16_224 | ImageNet(1M) | 59M | 76.44% |
| Mixer_L16 | timm | Mixer_largr_16_224 | ImageNet(1M) | 207M | 71.76% |
| | | | | | |
| Beit_B16 | timm | Beit_base_16_224 | ImageNet21k(14M) | 87M | 85.2% | https://github.com/microsoft/unilm/tree/master/beit
| Beit_L16 | timm | Beit_large_16_224 | ImageNet21k(14M) | 304M | 87.4% | https://github.com/microsoft/unilm/tree/master/beit
| | | | | | |
| Deit_B16 | timm | Deit_base_16_224 | ImageNet(1M) | 86M | 83.4% | https://github.com/facebookresearch/deit
| Deit_S16 | timm | Deit_small_16_224 | ImageNet(1M) | 22M | 81.2% | https://github.com/facebookresearch/deit
| | | | | | |
| EffN_b7_NS | timm | Efficientnet_b7_noisy_student | JFT(300M) | 66M | 86.9% | efficientnet#2-using-pretrained-efficientnet-checkpoints
| EffN_l2_NS | timm | Efficientnet_l2_noisy_student |  JFT(300M) | 480M|  88.4% | efficientnet#2-using-pretrained-efficientnet-checkpoints -->


**Some Samples from the data:**

<p float="left">
  <img src="README images/airliner_roll_bg1_14.png" width="200" />
  <img src="README images/cannon_roll_bg1_116.png" width="200" /> 
  <img src="README images/tank_yaw_nobg_86.png" width="200" />
  <img src="README images/tractor_yaw_nobg_132.png" width="200" />
</p>
<p float="left">
  <img src="README images/barberchair_roll_bg1_2.png" width="200" />
  <img src="README images/fireengine_roll_bg1_6.png" width="200" /> 
  <img src="README images/shoppingcart_yaw_nobg_14.png" width="200" />
    <img src="README images/tablelamp_yaw_nobg_80.png" width="200" />
</p><p float="left">
  <img src="README images/parkbench_pitch_bg1_192.png" width="200" />
  <img src="README images/mountainbike_pitch_bg1_64.png" width="200" /> 
  <img src="README images/hammerhead_pitch_bg1_28.png" width="200" />
    <img src="README images/forklift_roll_bg1_0.png" width="200" />
</p><p float="left">
  <img src="README images/jeep_pitch_bg1_108.png" width="200" />
  <img src="README images/rockingchair_yaw_nobg_24.png" width="200" /> 
  <img src="README images/barberchair_roll_bg1_2.png" width="200" />
    <img src="README images/foldingchair_pitch_bg1_40.png" width="200" />
</p>

