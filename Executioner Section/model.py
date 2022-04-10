from torch import nn
import segmentation_models_pytorch as smp 
from segmentation_models_pytorch.losses import DiceLoss

#Now creating our segmentation model class
class SegmentingModel(nn.Module):
  def __init__(self,Encoder,weights) -> None:
      super(SegmentingModel,self).__init__()
      self.arc=smp.Unet(encoder_name=Encoder,
                        encoder_weights=weights,
                        in_channels=3,
                        classes=1,
                        activation=None
                        )
  def forward(self,image,mask=None):
    logits=self.arc(image)
    if mask!=None:
      loss1=DiceLoss(mode='binary')(logits,mask)
      loss2=nn.BCEWithLogitsLoss()(logits,mask)
      return logits,loss1+loss2
    return logits

#Now creating a training function
def training_function(dataloader,model,optimizer,DEVICE):
  model.train()
  train_loss=0.0
  for image,mask in dataloader:
    image=image.to(DEVICE)
    mask=mask.to(DEVICE)

    optimizer.zero_grad()
    logits,loss=model(image,mask)
    loss.backward()
    optimizer.step()
    train_loss+=loss.item()
  return train_loss/len(dataloader)

#Now creating a training function
def eval_function(dataloader,model,DEVICE):
  model.train()
  eval_loss=0.0
  for image,mask in dataloader:
    image=image.to(DEVICE)
    mask=mask.to(DEVICE)

    logits,loss=model(image,mask)
    eval_loss+=loss.item()
  return eval_loss/len(dataloader)