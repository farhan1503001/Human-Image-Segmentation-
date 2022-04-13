import torch
import torchvision
from dataset import CarDataset
from augmentations import *
from torch.utils.data.dataloader import DataLoader

#Now saving and loading checkpoints
def save_checkpoints(state,filename='my_checkpoint.pth.tar'):
    print("----Saving checkpoint---")
    torch.save(state,filename)
def load_checkpoints(checkpoint,model):
    print("----Loading Checkpoint----")
    torch.load(checkpoint['state_dict'])
    
def get_dataloader(traim_img_dir,train_mask_dir,
                   test_img_dir,test_mask_dir,
                   batch_size,
                   train_transformer,
                   test_transformer,
                   num_workers=4,
                   PIN_MEMORY=True,
                   ):
    
    train_dataset=CarDataset(traim_img_dir,train_mask_dir,train_transformer)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,
                            pin_memory=PIN_MEMORY,
                            num_workers=num_workers,
                            shuffle=True
                            )
    test_dataset=CarDataset(test_img_dir,test_mask_dir,test_transformer)
    test_loader=DataLoader(test_dataset,
                           batch_size=batch_size,
                           pin_memory=PIN_MEMORY,
                           num_workers=num_workers,
                           shuffle=False
                           )
    return train_loader,test_loader

def check_accuracy(loader,model,device='cuda'):
    num_correct=0
    num_pixels=0
    Dice_score=0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device).unsqueeze(1)
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()
            num_correct+=(preds==y).sum()
            num_pixels+=torch.numel(preds)
            Dice_score+=(2*(preds*y).sum())/((preds+y).sum()+1e-8)
    print(f"Got{num_correct}/{num_correct} pixels with accuracy {num_correct}/{num_pixels}*100.00:.2f")
    print(f"Dice score is : {Dice_score/len(loader)}")
    
    model.train()
    
def save_predictions_as_images(loader,model,folder='savedimages/',device='cuda'):
    model.eval()
    
    for idx,(x,y) in enumerate(loader):
        x=x.to(device)
        with torch.no_grad():
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()
            torchvision.utils.save_image(preds,f"{folder}_pred_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1),f"{folder}_pred_{idx}.png")
    model.train()