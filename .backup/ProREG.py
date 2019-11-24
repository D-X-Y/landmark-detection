# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import torch
import torch.nn as nn

def weights_init_reg(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    m.weight.data.fill_(1)
    m.bias.data.zero_()
  elif isinstance(m, nn.Linear):
    n = m.weight.size(1)
    m.weight.data.normal_(0, 0.01)
    m.bias.data.zero_()


class AddCoords(nn.Module):

  def __init__(self, with_r=False):
    super().__init__()
    self.with_r = with_r

  def forward(self, input_tensor):
    """
    Args:
      input_tensor: shape(batch, channel, x_dim, y_dim)
    """
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    xx_channel, yy_channel = xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)
    ret = torch.cat([
      input_tensor,
      xx_channel,
      yy_channel], dim=1)

    if self.with_r:
      rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
      ret = torch.cat([ret, rr], dim=1)

    return ret


class CoordConv(nn.Module):

  def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
    super().__init__()
    self.addcoords = AddCoords(with_r=with_r)
    in_size = in_channels+2
    if with_r:
      in_size += 1
    self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

  def forward(self, x):
    ret = self.addcoords(x)
    ret = self.conv(ret)
    return ret

def conv_bn(inp, oup, kernels, stride, pad):
  return nn.Sequential(
    nn.Conv2d(inp, oup, kernels, stride, pad, bias=False),
    nn.BatchNorm2d(oup),
    nn.ReLU6(inplace=True)
  )


def conv_1x1_bn(inp, oup):
  return nn.Sequential(
    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    nn.BatchNorm2d(oup),
    nn.ReLU6(inplace=True)
  )


class InvertedResidual(nn.Module):
  def __init__(self, inp, oup, stride, expand_ratio):
    super(InvertedResidual, self).__init__()
    self.stride = stride
    assert stride in [1, 2]

    hidden_dim = round(inp * expand_ratio)
    self.use_res_connect = self.stride == 1 and inp == oup

    if expand_ratio == 1:
      self.conv = nn.Sequential(
        # dw
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
      )
    else:
      self.conv = nn.Sequential(
        # pw
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
      )

  def forward(self, x):
    if self.use_res_connect:
      return x + self.conv(x)
    else:
      return self.conv(x)


class MobileNetV2REG(nn.Module):
  def __init__(self, input_dim, input_channel, width_mult, pts_num):
    super(MobileNetV2REG, self).__init__()
    self.pts_num = pts_num
    block = InvertedResidual
    interverted_residual_setting = [
      # t, c, n, s
      [1, 48 , 1, 1],
      [2, 48 , 5, 2],
      [2, 96 , 1, 2],
      [4, 96 , 6, 1],
      [2, 16 , 1, 1],
    ]

    input_channel = int(input_channel * width_mult)
    features      = [conv_bn(input_dim, input_channel, (3,3), 2, 1)]
    # building inverted residual blocks
    for t, c, n, s in interverted_residual_setting:
      output_channel = int(c * width_mult)
      for i in range(n):
        if i == 0: stride = s
        else     : stride = 1
        features.append( block(input_channel, output_channel, stride, expand_ratio=t) )
        input_channel = output_channel
    features.append( nn.AdaptiveAvgPool2d( (14,14) ) )
    self.features = nn.Sequential(*features)
  
    self.S1 = nn.Sequential(
                  CoordConv(input_channel  , input_channel*2, True, kernel_size=3, padding=1),
                  conv_bn(input_channel*2, input_channel*2, (3,3), 2, 1))
    self.S2 = nn.Sequential(
                  CoordConv(input_channel*2, input_channel*4, True, kernel_size=3, padding=1),
                  conv_bn(input_channel*4, input_channel*8, (7,7), 1, 0))

    output_neurons  = 14*14*input_channel + 7*7*input_channel*2 + input_channel*8
    self.locator    = nn.Sequential(
                         nn.Linear(output_neurons, pts_num*2))

    #self.classifier = nn.Linear(output_neurons, pts_num)
    #self.classifier = nn.Sequential(
    #                     block(input_channel*1, input_channel*4, 1, 2),
    #                     nn.AdaptiveAvgPool2d( (16,12) ),
    #                    block(input_channel*4, input_channel*4, 1, 2),
    #                     nn.AdaptiveAvgPool2d( (8,6) ),
    #                     nn.Conv2d(input_channel*4, pts_num, (8,6)))

    self.apply( weights_init_reg )

  def forward(self, x):
    batch, C, H, W = x.size()
    features = self.features(x)
    S1 = self.S1( features )
    S2 = self.S2( S1 )
    tensors = torch.cat((features.view(batch, -1), S1.view(batch, -1), S2.view(batch, -1)), dim=1)
    batch_locs = self.locator(tensors).view(batch, self.pts_num, 2)
    #batch_scos = self.classifier(tensors).view(batch, self.pts_num, 1)
    return batch_locs


if __name__ == '__main__':
  model = MobileNetV2REG(3, 24, 1, 18) # REG on AFLW
