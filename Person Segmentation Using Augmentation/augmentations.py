import albumentations as A 
from albumentations.pytorch import ToTensorV2
Image_height=256
Image_width=256
def train_augmentation():
    train_transform=A.Compose(
        [
            A.Resize(height=Image_height,width=Image_width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            A.Rotate(limit=35,p=1.0),
            ToTensorV2()
            
        ]
    )
    return train_transform

def valid_augmentation():
    validation_transform=A.Compose(
        [
            A.Resize(height=Image_height,width=Image_width),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    return validation_transform