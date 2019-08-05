import torch
import torch.nn as nn
from torch.nn import init

class TeacherNet(nn.Module):
  def __init__(self, input_dim, n_layers=3):
    super(TeacherNet, self).__init__()
    sequence = [
      nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.2, True)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2**n, 8)
      sequence += [
        nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
        nn.InstanceNorm2d(64 * nf_mult, affine=False),
        nn.LeakyReLU(0.2, True)
      ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence += [
      nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=4, stride=1, padding=1, bias=True),
      nn.InstanceNorm2d(64 * nf_mult, affine=False),
      nn.LeakyReLU(0.2, True)
    ]

    sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

    self.model = nn.Sequential(*sequence)

  def forward(self, inputs):
    return self.model(inputs)
