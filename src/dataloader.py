import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class CO3D_Dataset(Dataset):

    def __init__(self, root_dir, co3d_id_map, transform=None):
        '''
        root_dir/
                ├── class_x
                |     |
                |     images
                │       ├── xxx.png
                │       ├── xxy.png
                │       └── ...
                │       
                └── class_y
                      |
                      images
                        ├── 123.png
                        ├── nsdf3.png
                        └── ...
                    
        '''

        self.path_and_label = []
        self.targets = []
        for id, class_folder in enumerate( os.listdir(root_dir)):
            self.path_and_label += [(os.path.join(root_dir, class_folder, 'images', image_name), co3d_id_map[class_folder.lower()]) for image_name in os.listdir(os.path.join(root_dir, class_folder, 'images')) if not 'ipynb_checkpoints' in image_name]
        print(f'testing on {len(self.path_and_label)} images')
        self.transform = transform
        self.root_dir = root_dir
    def __len__(self):
        return len(self.path_and_label)

    def __getitem__(self, idx):
        '''
        Returns the PIL image and the image name
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.path_and_label[idx][0]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, self.path_and_label[idx][1]



class ObjectPoseDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.all_paths = [ os.path.join(*p.parts) for p in Path(root_dir).rglob('*.png') ]
  
        # self.all_paths = os.listdir(root_dir)
        self.transform = transform
        self.root_dir = root_dir
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        '''
        Returns the PIL image and the image name
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()



        img_path = self.all_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, self.all_paths[idx] 




class ImageNetV2Dataset(Dataset):

    def __init__(self, root_dir, transform=None):


      self.imagenetv2_paths = []

      for folder in os.listdir(f"{root_dir}"):
        if os.path.isdir((f"{root_dir}/{folder}")):
          for img in os.listdir(f"{root_dir}/{folder}"):
            self.imagenetv2_paths.append((f"{root_dir}/{folder}/{img}", int(folder)))

      self.transform = transform
      self.root = root_dir

    def pil_loader(self, path: str) -> Image.Image:
      # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
      with open(path, "rb") as f:
          img = Image.open(f)
          return img.convert("RGB")

    def __len__(self):
        return len(self.imagenetv2_paths)

    def __getitem__(self, idx):
        '''
        Returns the PIL image and the image name
        '''

        img_path, label = self.imagenetv2_paths[idx]
        image = self.pil_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

