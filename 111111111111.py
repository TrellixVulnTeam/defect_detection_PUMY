import numpy as np
import torch
import torch.nn as nn

anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
# print(len(anchors))
ls = np.zeros((4, 3, 3, 4), int)
# print(ls[::2, ..., ::2].shape)
x = torch.ones((5, 5, 5, 5))
implicit = nn.Parameter(torch.zeros(5, 5, 5, 5))
nn.init.normal_(implicit, mean=0, std=0.02)
x = x + implicit

print(torch.tensor(anchors).float().view(3, -1, 2).shape)
