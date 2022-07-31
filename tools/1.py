import torch

a=torch.randn(2,3,2,3)
b =  a.mean(1,keepdim = False)
print(b)