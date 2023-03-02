import torch

# 将 RGB 图像转换为 LAB 图像的自定义函数
class RGB2LAB(torch.nn.Module):
    def __init__(self):
        super(RGB2LAB, self).__init__()

    def forward(self, img):
        # 将 RGB 值转换为 0 到 1 之间的浮点数
        img = img.float() / 255.0

        # 将 RGB 图像转换为 XYZ 图像
        R, G, B = torch.unbind(img, dim=2)
        X = 0.412453*R + 0.357580*G + 0.180423*B
        Y = 0.212671*R + 0.715160*G + 0.072169*B
        Z = 0.019334*R + 0.119193*G + 0.950227*B
        XYZ = torch.stack((X, Y, Z), dim=2)

        # 将 XYZ 图像转换为 LAB 图像
        XYZ_ref = torch.tensor([0.950456, 1.0, 1.088754]).to(img.device)
        XYZ_normalized = XYZ / XYZ_ref
        epsilon = 0.008856
        kappa = 903.3
        f = lambda t: torch.where(t > epsilon, torch.pow(t, 1/3), (kappa*t + 16)/116)
        fx, fy, fz = [f(t) for t in torch.unbind(XYZ_normalized, dim=2)]
        L = 116*fy - 16
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        LAB = torch.stack((L, a, b), dim=2)

        # 将 L、A、B 通道的范围调整到 -1 到 1 之间
        LAB[..., 0] = LAB[..., 0] / 50.0 - 1.0
        LAB[..., 1:] = LAB[..., 1:] / 128.0

        return LAB


import torch

# 将 LAB 图像转换为 RGB 图像的自定义函数
class LAB2RGB(torch.nn.Module):
    def __init__(self):
        super(LAB2RGB, self).__init__()

    def forward(self, img):
        # 将 L、A、B 通道的范围调整回原始范围
        L, a, b = torch.unbind(img, dim=2)
        L = (L + 1.0) * 50.0
        a = a * 128.0
        b = b * 128.0
        LAB = torch.stack((L, a, b), dim=2)

        # 将 LAB 图像转换为 XYZ 图像
        epsilon = 0.008856
        kappa = 903.3
        f_inv = lambda t: torch.where(t > epsilon, torch.pow(t, 3), (116*t - 16)/kappa)
        L, a, b = torch.unbind(LAB, dim=2)
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        fx, fz = [f_inv(t) for t in (fx, fz)]
        X = 0.950456*fx + 0.000000*fz + 0.000970*fy
        Y = 1.000000*fy - 0.000070*fx - 0.000012*fz
        Z = 1.088754*fz + 0.072098*fy + 0.000000*fx
        XYZ = torch.stack((X, Y, Z), dim=2)

        # 将 XYZ 图像转换为 RGB 图像
        M = torch.tensor([
            [ 3.24048134, -1.53715152, -0.49853633],
            [-0.96925495,  1.87599001,  0.04155593],
            [ 0.05564664, -0.20404134,  1.05731107]
        ]).to(img.device)
        RGB = torch.matmul(XYZ, M.t())
        RGB = torch.where(RGB <= 0.0031308, 12.92*RGB, 1.055*torch.pow(RGB, 1/2.4) - 0.055)
        RGB = torch.clamp(RGB, 0.0, 1.0)

        # 将 RGB 图像转换为 0 到 255 的整数并转换为 PIL 图像
        RGB = (255.0*RGB).type(torch.uint8)
        RGB_pil = RGB.permute(2, 0, 1)

        return RGB_pil
