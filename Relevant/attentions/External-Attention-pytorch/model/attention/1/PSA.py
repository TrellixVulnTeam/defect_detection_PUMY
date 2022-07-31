import numpy as np
import torch
from torch import nn
from torch.nn import init

# 这是深大5月30日在arXiv上上传的一篇文章，本文的目的是如何获取并探索不同尺度的空间信息来丰富特征空间。
# 网络结构相对来说也比较简单，主要分成四步，第一部，将原来的feature根据通道分成n组然后对不同的组进行不同尺度的卷积，得到新的特征W1；
# 第二部，用SE在原来的特征上进行SE，从而获得不同的阿头疼托尼；第三部，对不同组进行SOFTMAX；第四部，将获得attention与原来的特征W1相乘。
# 作者：小番茄666丶 https://www.bilibili.com/read/cv11665879?spm_id_from=333.999.0.0 出处：bilibili

class PSA(nn.Module):

    def __init__(self, channel=512,reduction=4,S=4):
        super().__init__()
        self.S=S

        self.convs=[]
        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=2*(i+1)+1,padding=i+1))

        self.se_blocks=[]
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel//S, channel // (S*reduction),kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S*reduction), channel//S,kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax=nn.Softmax(dim=1)


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
        b, c, h, w = x.size()

        #Step1:SPC module
        SPC_out=x.view(b,self.S,c//self.S,h,w) #bs,s,ci,h,w
        for idx,conv in enumerate(self.convs):
            SPC_out[:,idx,:,:,:]=conv(SPC_out[:,idx,:,:,:])

        #Step2:SE weight
        se_out=[]
        for idx,se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:,idx,:,:,:]))
        SE_out=torch.stack(se_out,dim=1)
        SE_out=SE_out.expand_as(SPC_out)

        #Step3:Softmax
        softmax_out=self.softmax(SE_out)

        #Step4:SPA
        PSA_out=SPC_out*softmax_out
        PSA_out=PSA_out.view(b,-1,h,w)

        return PSA_out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    psa = PSA(channel=512,reduction=8)
    output=psa(input)
    a=output.view(-1).sum()
    a.backward()
    print(output.shape)

    