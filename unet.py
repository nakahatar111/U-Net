import torch
import torch.nn as nn

def convolution(in_channel, out_channel):
  return nn.Sequential(
    nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
  )

def convolution_down(in_channel):
  return nn.Sequential(
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.Conv2d(2*in_channel, 2*in_channel, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
  )

def convolution_bottleneck(in_channel):
  return nn.Sequential(
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.Conv2d(2*in_channel, 2*in_channel, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(2*in_channel, in_channel, kernel_size=2, stride=2, padding=0),
  )

def convolution_up(in_channel):
  return nn.Sequential(
    nn.Conv2d(in_channel, in_channel//2, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, stride=1, padding=0),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(in_channel//2, in_channel//4, kernel_size=2, stride=2, padding=0),
  )

def cross_connection(x, skip_connection):
    h = (skip_connection.shape[2] - x.shape[2])//2
    w = (skip_connection.shape[3] - x.shape[3])//2
    return torch.cat((x, skip_connection[:, :, h:h + x.shape[2], w:w + x.shape[3]]), dim=1)

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolution_start = convolution(3, 64)
    self.convolution_down_1 = convolution_down(64)
    self.convolution_down_2 = convolution_down(128)
    self.convolution_down_3 = convolution_down(256)
    self.bottle_neck = convolution_bottleneck(512)
    self.convolution_up_3 = convolution_up(1024)
    self.convolution_up_2 = convolution_up(512)
    self.convolution_up_1 = convolution_up(256)
    self.convolution_block = convolution(128, 64)
    self.convolution_end = nn.Conv2d(64, 12, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    x1 = self.convolution_start(x)
    x2 = self.convolution_down_1(x1)
    x3 = self.convolution_down_2(x2)
    x4 = self.convolution_down_3(x3)
    x = self.bottle_neck(x4)
    x = cross_connection(x, x4)
    x = self.convolution_up_3(x)
    x = cross_connection(x, x3)
    x = self.convolution_up_2(x)
    x = cross_connection(x, x2)
    x = self.convolution_up_1(x)
    x = cross_connection(x, x1)
    x = self.convolution_block(x)
    return self.convolution_end(x)