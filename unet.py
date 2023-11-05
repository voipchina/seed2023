import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    连续两次使用3*3卷积，在原有Unet基础上加入BN
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样部分
    Maxpooling => DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样部分
    双线性插值/反卷积 + 特征融合 => DoubleConv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        :param x1: 要进行上采样的特征图
        :param x2: 要融合的特征图（来自下采样阶段）
        :return: 融合后的特征
        """

        x1 = self.up(x1)
        # 输入格式为C * H * W(Batch * C * H * W)
        # H之差
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # W之差
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # 将x1padding至x2大小
        # pad二维扩充时参数顺序为左右上下
        # //代表除以并保留整数，也可以理解为向下取整或取商不取余
        x1 = F.pad(x1, pad=[diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # (Batch * C * H * W)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出层卷积
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),

        )

    def forward(self, x):
        return self.out_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)    # 进行二分类


if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    input = torch.rand(2, 1, 224, 224)
    pred = net(input)
    print(pred.shape)

