import cv2
import numpy as np
import torch
from augmentation import training_augment,validation_augment
from torch.utils.data.dataset import Dataset

#Now creating segmentation dataset
class SegmentarSet(Dataset):
  def __init__(self,df,augmentations):
    self.dataframe=df
    self.augmentations=augmentations
  def __len__(self):
    return len(self.dataframe)
  def __getitem__(self, index):
    row=self.dataframe.iloc[index]

    image_path=row.images
    mask_path=row.masks
    #Now reading
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    mask=np.expand_dims(mask,axis=-1)

    if self.augmentations:
      data=self.augmentations(image=image,mask=mask)
      image=data['image']
      mask=data['mask']
    #Now reshaping for pytorch
    image=np.transpose(image,(2,0,1)).astype(np.float32)
    mask=np.transpose(mask,(2,0,1)).astype(np.float32)
    #Now we are converting image to tensor
    image=torch.Tensor(image)/255.0
    mask=torch.round(torch.Tensor(mask)/255.0)

    return image,mask