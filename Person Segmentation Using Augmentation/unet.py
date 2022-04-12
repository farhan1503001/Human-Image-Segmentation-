from turtle import forward
import torch
import torch.nn as nn

#Now we will create a Double Conv Custom layer for this model
class DoubleConv(nn.Module):
    def __init__(self,in_channles,out_channles) -> None:
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channles,out_channles,3,1,1,bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channles,out_channles,3,1,1,bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

#Main Unet class we are writing
class UNET(nn.Module):
    def __init__(self,in_channels,out_channles=1,features=[64,128,256,512]) -> None:
        super(UNET,self).__init__()
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
        self.bottlenecks=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        
        #Downsampling part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels=feature
        
        #Upsampling part of UNET where we will use Transpose Conv
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
                )
            self.ups.append(DoubleConv(feature*2,feature))
            
        #Now we will write a bottleneck layer at the very low of U structure
        self.bottlenecks=DoubleConv(features[-1],features[-1]*2)
        
        #Now our final classification layer
        self.fina_conv=nn.Conv2d(features[0],out_channles,kernel_size=1)
    def forward(self,x):
        #Now this function will run the model
        skip_connections=[]
        #Now first going on downsampling
        for layer in self.downs:
            x=layer(x)
            skip_connections.append(x)
            x=self.pool(x)
        #Now lower downsampling
        x=self.bottlenecks(x)
        #Now turning skip connections backwards
        skip_connections=skip_connections[::-1]
        #Now we will move towards up sampling
        for index in range(0,len(self.ups),2):
            x=self.ups[index](x)
            connection=skip_connections[index//2]
            #Now concatenate
            concat=torch.cat((connection,x),axis=1)
            x=self.ups[index+1](concat)
        return self.fina_conv(x)
    
    
if __name__ == '__main__':
    x=torch.randn((3,1,128,128))
    unet_model=UNET(in_channels=1,out_channles=1)
    predict=unet_model(x)
    print("Previously input size is ",x.shape)
    print("Here torch size is",predict.shape)