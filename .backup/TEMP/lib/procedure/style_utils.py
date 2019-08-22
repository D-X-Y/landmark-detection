import torch

def generate_noise(inputs):
  assert inputs.dim() == 4, 'The input size is wrong : {:}'.format(inputs.size())
  batch, C, H, W = inputs.size()
  noise = torch.randn((batch, 1, H, 1))
  noise = noise.repeat(1, 1, 1, W)
  noise = noise.to(inputs.device)
  final = torch.cat((inputs, noise), dim=1)
  return final
  
def test_noise():
  inputs = torch.randn((128, 3, 32, 32))
  noises = generate_noise(inputs)
  print ('noise size : {:}'.format(noises.size()))
#test_noise()
