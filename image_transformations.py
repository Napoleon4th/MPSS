from PIL import Image, ImageOps
import torchvision.transforms.functional as F

def flip_left_right(image):
    #MR1: Flip left-right
    return ImageOps.mirror(image)

def flip_up_down(image):
    #MR2: Flip up-down
    return ImageOps.flip(image)

def rotate_left(image, angle=5):
    #MR3: Rotate left five degrees
    return image.rotate(angle, resample=Image.BILINEAR)

def rotate_right(image, angle=5):
    #MR4: Rotate right five degrees
    return image.rotate(-angle, resample=Image.BILINEAR)

def shear_left(image, shear=5):
    #MR5: Shear left five degrees
    shear_transform = F.affine(image, angle=0, translate=(0, 0), scale=1, shear=(-shear, 0))
    return shear_transform