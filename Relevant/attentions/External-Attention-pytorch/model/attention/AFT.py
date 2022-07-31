import numpy as np
import torch
from torch import nn
from torch.nn import init


# 这是苹果团队2021年6月16日在arXiv上发布的工作，主要工作是简化Self-Attention。
#
# Transformer近几年被用于各种任务中，但是由于Self-Attention的与输入数据大小呈平方关系的时间和空间复杂度，它不能被用于太大的数据中。近几年，基于简化SA的复杂度，很多工作也被提出：稀疏注意力、局部哈希、低质分解...
#
# 本文提出了一个Attention Free Transformer（AFT），AFT也是由QKV三部分组成，不同的是QK不是做点积。而是将KV直接融合了，从而来保证对应位置的交互，然后Q与融合后的特征进行了对应位置相乘，来减少计算量。
#
# 总体上原理跟Self-Attention相似，不同的是Self-Attention用的是点积，而这里用的是对应位置相乘，所以大大减少了计算量。

class AFT_FULL(nn.Module):

    def __init__(self, d_model,n=49,simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model,d_model)
        if(simple):
            self.position_biases=torch.zeros((n,n))
        else:
            self.position_biases=nn.Parameter(torch.ones((n,n)))
        self.d_model = d_model
        self.n=n
        self.sigmoid=nn.Sigmoid()

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

    def forward(self, input):

        bs, n,dim = input.shape

        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input).view(1,bs,n,dim) #1,bs,n,dim
        v = self.fc_v(input).view(1,bs,n,dim) #1,bs,n,dim
        
        numerator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1))*v,dim=2) #n,bs,dim
        denominator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1)),dim=2) #n,bs,dim

        out=(numerator/denominator) #n,bs,dim
        out=self.sigmoid(q)*(out.permute(1,0,2)) #bs,n,dim

        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    aft_full = AFT_FULL(d_model=512, n=49)
    output=aft_full(input)
    print(output.shape)

    