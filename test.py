
from attacks.color_space_convert import RGB2LAB, LAB2RGB
from torchvision import transforms
from PIL import Image
import torch


rgb2lab = RGB2LAB()
lab2rgb = LAB2RGB()

def color_transfer(sc, tc):
    sc = rgb2lab(sc)
    s_mean = torch.mean(sc, dim=[0,1], keepdim=True)
    s_std = torch.std(sc, dim=[0,1], keepdim=True)

    tc = rgb2lab(tc)
    t_mean = torch.mean(tc, dim=[0,1], keepdim=True)
    t_std = torch.std(tc, dim=[0,1], keepdim=True)

    img_n = ((sc-s_mean)*(t_std/s_std)) +t_mean 
    img_n = torch.clip(img_n, max=255, min=0)
    dst = lab2rgb(img_n)
    return dst




image = Image.open("m01.JPEG")
target = Image.open("m02.JPEG")

trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()])

image = trans(image)
target = trans(target)

image_ = color_transfer(target,image)

image_ = transforms.ToPILImage()(image_)
image_.save("m03.jpg")