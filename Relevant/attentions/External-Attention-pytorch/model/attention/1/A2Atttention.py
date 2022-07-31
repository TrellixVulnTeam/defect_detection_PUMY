import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
#
# 这是NeurIPS2018上的一篇文章，这篇论文主要是做空间注意力的。并且这篇文章的方法跟做法跟self-attention非常相似，但是包装上就比较“花里胡哨”。
#
# input用1x1的卷积变成A，B，V（类似self-attention的Q，K，V）。本文的方法主要分为两个步骤，第一步，feature gathering，首先用A和B进行点乘，得到一个聚合全局信息的attention，标记为G。然后用G和V进行点乘，得到二阶的attention。（个人觉得这个有点像Attention on Attention（AOA）[11]，ICCV2019的那篇文章）。
#
# 从实验结果上看，这个结构的效果还是非常不错的，作者在分类（ImageNet）和行为识别（Kinetics ， UCF-101）任务上做了实验，都取得非常好的效果，相比于Non-Local[12]、SENet[13]等模型，都有不错的提升。

class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ 


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    a2 = DoubleAttention(512,128,128,True)
    output=a2(input)
    print(output.shape)

    