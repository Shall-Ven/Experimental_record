import torch

# 将 RGB 图像转换为 LAB 图像的自定义函数
class RGB2LAB(torch.nn.Module):
    def __init__(self):
        super(RGB2LAB, self).__init__()

    def forward(self, img):
    #srgb = check_image(srgb)
        img = torch.permute(img,(1,2,0))
        srgb_pixels = img.reshape([-1,3])#tf.reshape(srgb, [-1, 3])

        linear_mask = torch.as_tensor(srgb_pixels <= 0.04045, dtype=torch.float32)
        exponential_mask = torch.as_tensor(srgb_pixels > 0.04045, dtype=torch.float32)
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        rgb_to_xyz = torch.as_tensor([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334], # R
            [0.357580, 0.715160, 0.119193], # G
            [0.180423, 0.072169, 0.950227], # B
        ])
        xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz)

        xyz_normalized_pixels = torch.multiply(xyz_pixels, torch.as_tensor([1/0.950456, 1.0, 1/1.088754]))

        epsilon = 6/29
        linear_mask = torch.as_tensor(xyz_normalized_pixels <= (epsilon**3), dtype=torch.float32)
        exponential_mask = torch.as_tensor(xyz_normalized_pixels > (epsilon**3), dtype=torch.float32)
        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

        # convert to lab
        fxfyfz_to_lab = torch.as_tensor([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
        ])
        lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + torch.as_tensor([-16.0, 0.0, 0.0])

        return lab_pixels.reshape(img.shape)


import torch

# 将 LAB 图像转换为 RGB 图像的自定义函数
class LAB2RGB(torch.nn.Module):
    def __init__(self):
        super(LAB2RGB, self).__init__()

    def forward(self, lab):
        lab_pixels = lab.reshape([-1, 3])
                # convert to fxfyfz
        lab_to_fxfyfz = torch.as_tensor([
            #   fx      fy        fz
            [1/116.0, 1/116.0,  1/116.0], # l
            [1/500.0,     0.0,      0.0], # a
            [    0.0,     0.0, -1/200.0], # b
        ])
        fxfyfz_pixels = torch.matmul(lab_pixels + torch.as_tensor([16.0, 0.0, 0.0]), lab_to_fxfyfz)

        # convert to xyz
        epsilon = 6/29
        linear_mask = torch.as_tensor(fxfyfz_pixels <= epsilon, dtype=torch.float32)
        exponential_mask = torch.as_tensor(fxfyfz_pixels > epsilon, dtype=torch.float32)
        xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

        # denormalize for D65 white point
        xyz_pixels = torch.multiply(xyz_pixels, torch.as_tensor([0.950456, 1.0, 1.088754]))

        xyz_to_rgb = torch.as_tensor([
            #     r           g          b
            [ 3.2404542, -0.9692660,  0.0556434], # x
            [-1.5371385,  1.8760108, -0.2040259], # y
            [-0.4985314,  0.0415560,  1.0572252], # z
        ])
        rgb_pixels = torch.matmul(xyz_pixels, xyz_to_rgb)
        # avoid a slightly negative number messing up the conversion
        rgb_pixels = torch.clip(rgb_pixels, 0.0, 1.0)
        linear_mask = torch.as_tensor(rgb_pixels <= 0.0031308,  dtype=torch.float32)
        exponential_mask = torch.as_tensor(rgb_pixels > 0.0031308, dtype=torch.float32)
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return torch.permute(srgb_pixels.view(lab.shape), (2,0,1))

