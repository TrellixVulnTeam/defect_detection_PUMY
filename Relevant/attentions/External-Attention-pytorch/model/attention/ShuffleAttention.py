import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter



# 这是南大在ICASSP 2021发表的一篇论文，这篇文章同样是捕获两个注意力：通道注意力和空间注意力。本文提出的ShuffleAttention主要分为三步：
#
# 1.首先将输入的特征分为组，然后每一组的特征进行split，分成两个分支，分别计算 channel attention 和 spatial attention，两种 attention 都使用可训练的参数（当时看结构图的时候，以为是这里是用了FC，但是读了源码之后，才发现是为每一个通道创建了一组可学习的参数） + sigmoid 的方法计算。
#
# 2.接着，两个分支的结果concat到一起，然后合并，得到和输入尺寸一致的 feature map。
#
# 3.最后，用一个 shuffle 层进行通道Shuffle（类似ShuffleNet[2]）。
#
# 作者在分类数据集 ImageNet-1K 和目标检测数据集 MS COCO 以及实例分割任务上做了实验，表明 SA 的性能要超过目前 SOTA 的方法，实现了更高的准确率，而且模型复杂度较低。

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


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


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = ShuffleAttention(channel=512,G=8)
    output=se(input)
    print(output.shape)

    