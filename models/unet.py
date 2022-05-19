import torch
import torch.nn as nn
import torch.nn.functional as F


# This unet is modified for range image (rectangular)
# Downsampling and upsampling is only done in horizontal direction
#

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1,2), mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=17, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # B,C,H,W  =  1,5,64,1024
        x1 = self.inc(x) # 1,64,64,1024
        x2 = self.down1(x1) # 1,128,64,512
        x3 = self.down2(x2) # # 1,256,64,256
        x4 = self.down3(x3) # 1,512,64,128
        x5 = self.down4(x4) # 1,512,64,64
        x = self.up1(x5, x4) # 1,256,64,128
        x = self.up2(x, x3) # 1,128,64,256
        x = self.up3(x, x2) # 1,64,64,512
        x = self.up4(x, x1) # 1,64,64,1024
        logits = self.outc(x) # 1,18,64,1024
        return logits

# import numpy as np
# net=UNet(n_channels=5,n_classes=18).double()
# x = np.random.random((1,5,64,1024)).astype(np.float32)
# x =torch.tensor(x).cuda()
# x= x.double()
# x = x.type(torch.DoubleTensor)
#
#
# input = np.arange(0,84).reshape(6,-1)
# arr3D = np.repeat(input[None,...],3,axis=0).astype(np.double)
# arr4D = np.repeat(arr3D[None,...],5,axis=0)

# maxpool = nn.MaxPool2d((1,2),stride=(1,2))
# x1 =maxpool(torch.tensor(x)).numpy()
# x2 =maxpool(torch.tensor(x1)).numpy()
# x3 =maxpool(torch.tensor(x2)).numpy()

# inc = DoubleConv(3, 64).double()
# res = inc(torch.tensor(arr4D))

# out = net(x)
# print("testig")