import torch

inp = torch.rand([1, 3, 224, 224], dtype=torch.float32)
vgg11 = VGG('vgg_11')
print(vgg11(inp).shape)
