import torch 
import albumentations as A 
import numpy as np
import torch.nn as nn
import tqdm
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from unet import UNET
from augmentations import *
from utility import *
#Now defining important hyperparameters
Learning_rate=1e-4
Batch_size=16
DEVICE='cuda'
Num_Epochs=10
Num_works=2
Image_height=256
Image_width=256
Pin_Memory=True
train_img_dir='data/train_images/'
train_mask_dir='data/train_masks/'
test_img_dir='data/test_images/'
test_mask_dir='data/test_masks/'
#Now writing a general training function
def train_function(dataloader,model,loss_func,optimizer,scaler):
    #starting single epoch
    loop=tqdm(dataloader)
    
    for index,(img,mask) in enumerate(loop):
        #setting them to cuda
        img=img.to(DEVICE)
        mask=mask.float().unsqueeze(1).to(DEVICE)
        #Forward propagation
        with torch.cuda.amp.autocast():
            logits=model(img)
            loss=loss_func(logits,mask)
        #Now backward propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #Now update tqdm
        loop.set_postfix(loss.item())
        
#Now running the final trainer function
def trainer():
    train_transform=train_augmentation()
    validation_transform=valid_augmentation()
    
    #Now we are calling the model
    model=UNET(in_channels=3,out_channles=1).to(DEVICE)
    #Now defining loss function
    loss_function=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters,lr=Learning_rate)
    
    train_loader,test_loader=get_dataloader(train_img_dir,
                                            train_mask_dir,test_img_dir,test_mask_dir,
                                            Batch_size,train_transform,validation_transform,Num_works,Pin_Memory
                                            )
    
    scaler=torch.cuda.amp.GradScaler()
    
    for epoch in range(Num_Epochs):
        train_function(train_loader,model,loss_function,optimizer,scaler)
        
        #save checkpoint
        checkpoint={
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict()
        }
        save_checkpoints(checkpoint)
        #checking accuracy
        check_accuracy(test_loader,model,DEVICE)
        #saving test images
        save_predictions_as_images(loader=test_loader,model=model,device=DEVICE)
        

if __name__ == '__main__':
    trainer()