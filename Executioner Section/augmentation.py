import albumentations as augmentar
IMAGE_SIZE=320


def training_augment():
  return augmentar.Compose(
      [
       augmentar.Resize(IMAGE_SIZE,IMAGE_SIZE),
       augmentar.HorizontalFlip(p=0.5),
       augmentar.VerticalFlip(p=0.5)
      ]
  )
def validation_augment():
  return augmentar.Compose(
      [
       augmentar.Resize(IMAGE_SIZE,IMAGE_SIZE)
      ]
  )