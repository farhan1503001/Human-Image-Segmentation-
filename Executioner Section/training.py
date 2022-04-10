from Custom_dataset import SegmentarSet
from augmentation import training_augment,validation_augment
from model import SegmentingModel,training_function,eval_function
import torch 
import cv2
import helper
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#Running necessary libraries

if __name__ == '__main__':
    CSV_FILE='Human-Segmentation-Dataset-master/train.csv'
    data_dir='/'
    IMAGE_SIZE=320
    Encoder='timm-efficientnet-b0'#change it as you like
    weights='imagenet'
    Batch_size=16
    DEVICE='cuda'
    #Reading image paths
    data=pd.read_csv(CSV_FILE)
    data.head()
    #Now finally creating our custom dataset
    train_frame,test_frame=train_test_split(data,test_size=0.2,random_state=42)
    trainset=SegmentarSet(train_frame,training_augment())
    validset=SegmentarSet(test_frame,validation_augment())
    
    print(f"Size of Trainset : {len(trainset)}")
    print(f"Size of Validset : {len(validset)}")
    
    #Creating data loader
    
    traindata_loader=DataLoader(trainset,batch_size=Batch_size,shuffle=True)
    validdata_loader=DataLoader(validset,batch_size=Batch_size,shuffle=False)
    print(f"Number of Batches in Train Loader {len(traindata_loader)}")
    print(f"Number of Batches in Test Loader {len(validdata_loader)}")
    #Now observing single batch shape
    #Now just seeing the size of first batch
    for image,mask in traindata_loader:
        break
    print(f"One batch Image shape: {image.shape}")
    print(f"One batch mask shape: {mask.shape}")
    #Start running model
    torch.cuda.empty_cache()
    model=SegmentingModel(Encoder=Encoder,weights=weights)
    model.to(DEVICE)
    #Now defining optimizer
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    EPOCHS=30#change as you like
    best_validation_loss=np.Inf
    for i in range(EPOCHS):
        train_loss=training_function(traindata_loader,model,optimizer)
        valid_loss=eval_function(validdata_loader,model)
        if best_validation_loss>valid_loss:
            torch.save(model.state_dict(),"Best_Model+EFF.pt")
            print("Model Saved")
            best_validation_loss=valid_loss
    print(f"Epoch {i+1} Train Loss {train_loss} Validation Loss {valid_loss}")
    index=22
    image,mask=validset[index]
    #Now loading the model
    model.load_state_dict(torch.load('/content/Best_Model+EFF.pt'))
    #Now predicting
    logits_mask=model(image.to(DEVICE).unsqueeze(0))
    pred_mask=torch.sigmoid(logits_mask)
    pred_mask=(pred_mask>0.5)*1.0
    
    helper.show_image(image,mask,pred_mask.detach().cpu().squeeze(0))
        