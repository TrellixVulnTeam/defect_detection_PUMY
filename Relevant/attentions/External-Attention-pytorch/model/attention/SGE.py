import numpy as np
import torch
from torch import nn
from torch.nn import init

# 这篇文章是SKNet[7]作者在19年的时候在arXiv上挂出的文章，是一个轻量级Attention的工作，从下面的核心代码中，可以看出，引入的参数真的非常少，self.weight和self.bias都是和groups呈一个数量级的（几乎就是常数级别）。
#
# 这篇文章的核心点是用局部信息和全局信息的相似性来指导语义特征的增强，总体的操作可以分为以下几步：
#
# 1）将特征分组，每组feature在空间上与其global pooling后的feature做点积（相似性）得到初始的attention mask
#
# 2）对该attention mask进行减均值除标准差的normalize，并同时每个group学习两个缩放偏移参数使得normalize操作可被还原
#
# 3）最后经过sigmoid得到最终的attention mask并对原始feature group中的每个位置的feature进行scale
#
# 实验部分，作者也是在分类任务（ImageNet）和检测任务（COCO）上做了实验，能够在比SK[7]、CBAM[8]、BAM[9]等网络参数和计算量更小的情况下，获得更好的性能，证明了本文方法的高效性。
# 每个bottleneck最后一个BN层之后，同时group设为64

class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
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
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    sge = SpatialGroupEnhance(groups=8)
    output=sge(input)
    print(output.shape)

    