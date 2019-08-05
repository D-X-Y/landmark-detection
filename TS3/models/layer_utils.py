import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
  def __init__(self, numIn, numOut):
    super(Residual, self).__init__()
    self.numIn = numIn
    self.numOut = numOut
    middle = self.numOut // 2

    self.conv_A = nn.Sequential(
                    nn.BatchNorm2d(numIn), nn.ReLU(inplace=True),
                    nn.Conv2d(numIn, middle, kernel_size=1, dilation=1, padding=0, bias=True))

    self.conv_B = nn.Sequential(
                    nn.BatchNorm2d(middle), nn.ReLU(inplace=True),
                    nn.Conv2d(middle, middle, kernel_size=3, dilation=1, padding=1, bias=True))

    self.conv_C = nn.Sequential(
                    nn.BatchNorm2d(middle), nn.ReLU(inplace=True),
                    nn.Conv2d(middle, numOut, kernel_size=1, dilation=1, padding=0, bias=True))

    if self.numIn != self.numOut:
      self.branch = nn.Sequential(
                      nn.BatchNorm2d(self.numIn), nn.ReLU(inplace=True),
                      nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

  def forward(self, x):
    residual = x
    
    main = self.conv_A(x)
    main = self.conv_B(main)
    main = self.conv_C(main)
    if hasattr(self, 'branch'):
      residual = self.branch( residual )
  
    return main + residual
