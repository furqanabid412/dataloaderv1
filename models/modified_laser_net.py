import numpy as np
import torch
from torch import nn
import time
import torch.nn.functional as F
from tqdm import tqdm

__all__ = ["LaserNet"]

BatchNorm = nn.BatchNorm2d
'''Basic Resnet Block
Conv2d
BN
Relu
Conv2d
BN
Relu

with residual connection between input and output
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.stride = stride
        self.project = None
        if in_channels!=out_channels:
            self.project = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.project != None:
            residual = self.project(residual)
        out += residual
        out = self.relu(out)

        return out

'''
Deconvolution Layer for upsampling
TransposeConv2d
BN
Relu
'''
class Deconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=(1,2),stride=(1,2),padding=0)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


'''
Feature Aggregator Module described in the LaserNet paper
'''
class Feature_Aggregator(nn.Module):
    def __init__(self,in_channels_1,in_channels_2,out_channels):
        super().__init__()
        self.deconv = Deconv(in_channels_2,out_channels)
        self.block_1 = BasicBlock(in_channels_1+in_channels_2,out_channels)
        self.block_2 = BasicBlock(out_channels,out_channels)

    def forward(self,x1,x2):
        x2 = self.deconv(x2)
        x1 = torch.cat([x1,x2],1)
        x1 = self.block_1(x1)
        x1 = self.block_2(x1)
        return x1

'''
DownSample module using Conv2d with stride > 1
Conv2d(stride>1)
BN
Relu
Conv2d
BN
Relu
'''
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=(1,2), padding=dilation, bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=dilation,bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.project = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=(1,2),padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        # x : B,C,H,W
        out = self.conv1(x) # outputs : B,C,H/2,W/2
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.project(residual)
        out += residual
        out = self.relu(out)
        return out

'''
Feature Extrator module described in LaserNet paper
DownSample input if not 1a
'''
class Feature_Extractor(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks=6,down_sample_input=False):
        super().__init__()
        self.down_sample = None
        self.down_sample_input = down_sample_input
        if down_sample_input:
            self.down_sample = DownSample(in_channels,out_channels)

        blocks_modules = []
        for i in range(num_blocks):
            if i == 0 and not down_sample_input:
                blocks_modules.append(BasicBlock(in_channels,out_channels))
            else:
                blocks_modules.append(BasicBlock(out_channels,out_channels))
        self.blocks = nn.Sequential(*blocks_modules)

    def forward(self,x):
        if self.down_sample_input:
            x = self.down_sample(x)
        x = self.blocks(x)
        return x

'''
Main Deep Aggregation class described as in LaserNet paper
num_outputs is the number of channels of the output image
output image has the same width and height as input image
LaserNet
'''

class LaserNet(nn.Module):
    def __init__(self,num_inputs, channels,num_outputs):
        super().__init__()
        # num_inputs=64
        # channels=[64,128,128]
        # num_outputs = 4

        # self.input_bn = BatchNorm(num_inputs)

        self.extract_1a = Feature_Extractor(num_inputs,channels[0]) # 64,64
        self.extract_2a = Feature_Extractor(channels[0],channels[1],down_sample_input=True) # 64,128
        self.extract_3a = Feature_Extractor(channels[1],channels[2],down_sample_input=True) # 128,128
        self.aggregate_1b = Feature_Aggregator(channels[0],channels[1],channels[1]) # 64,128,128
        self.aggregate_1c = Feature_Aggregator(channels[1],channels[2],channels[2]) # 128,128,128
        self.aggregate_2b = Feature_Aggregator(channels[1],channels[2],channels[2]) # 128,128,128
        self.conv_1x1 = nn.Conv2d(channels[2],num_outputs,kernel_size=1,stride=1) # 128,4

    def forward(self,x):
        # batch normalizing the input
        # x=self.input_bn(x)

        # Network
        x_1a = self.extract_1a(x.type(torch.float32))
        x_2a = self.extract_2a(x_1a)
        x_3a = self.extract_3a(x_2a)
        x_1b = self.aggregate_1b(x_1a,x_2a)
        x_2b = self.aggregate_2b(x_2a,x_3a)
        x_1c = self.aggregate_1c(x_1b,x_2b)
        out = self.conv_1x1(x_1c)
        return out


import numpy as np
net=LaserNet(5,[64,128,128],4)
input = np.random.random((6,5,64,1024))
out = net(torch.tensor(input))
print("testig")