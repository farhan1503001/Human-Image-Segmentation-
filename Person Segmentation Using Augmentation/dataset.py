import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from PIL import Image

class CarDataset(Dataset):
    def __init__(self,image_directory,mask_directory,transformation=None) -> None:
        self.image_directory=image_directory
        self.mask_directory=mask_directory
        self.transformation=transformation
        self.images=os.listdir(self.image_directory)
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index:int):
        img_path=os.path.join(self.image_directory,self.images[index])
        mask_path=os.path.join(self.mask_directory,self.images[index].replace('.jpg','_mask.gif'))
        #Now reading image using PIL
        image=np.array(Image.open(img_path).convert('RGB'))
        mask=np.array(Image.open(mask_path).convert('L'))
        #Now labelizing image
        mask[mask==255.0]=1.0
        
        if self.transformation is not None:
            augmentation=self.transformation(image=image,mask=mask)
            image=augmentation['image']
            mask=augmentation['mask']
        
        return image,mask