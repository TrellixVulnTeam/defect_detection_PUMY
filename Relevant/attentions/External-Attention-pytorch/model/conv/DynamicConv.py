import torch
from torch import nn
from torch.nn import functional as F


# class Attention(nn.Module):
#     def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.temprature = temprature
#         assert in_planes > ratio
#         hidden_planes = in_planes // ratio
#         self.net = nn.Sequential(
#             nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
#         )
#
#         if (init_weight):
#             self._initialize_weights()
#
#     def update_temprature(self):
#         if (self.temprature > 1):
#             self.temprature -= 1
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         att = self.avgpool(x)  # bs,dim,1,1
#         att = self.net(att).view(x.shape[0], -1)  # bs,K
#         return F.softmax(att / self.temprature, -1)
#
#
# class DynamicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
#                  temprature=30, ratio=4, init_weight=True):
#         super().__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = grounps
#         self.bias = bias
#         self.K = K
#         self.init_weight = init_weight
#         self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
#                                    init_weight=init_weight)
#
#         self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
#                                    requires_grad=True)
#         if (bias):
#             self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
#         else:
#             self.bias = None
#
#         if (self.init_weight):
#             self._initialize_weights()
#
#         # TODO 初始化
#
#     def _initialize_weights(self):
#         for i in range(self.K):
#             nn.init.kaiming_uniform_(self.weight[i])
#
#     def forward(self, x):
#         bs, in_planels, h, w = x.shape
#         softmax_att = self.attention(x)  # bs,K
#         x = x.view(1, -1, h, w)
#         weight = self.weight.view(self.K, -1)  # K,-1
#         aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
#                                                               self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k
#
#         if (self.bias is not None):
#             bias = self.bias.view(self.K, -1)  # K,out_p
#             aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
#             output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
#                               groups=self.groups * bs, dilation=self.dilation)
#         else:
#             output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                               groups=self.groups * bs, dilation=self.dilation)
#         h = int(h / self.stride)
#         w = int(w / self.stride)
#         output = output.view(bs, self.out_planes, h, w)
#         return output


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


if __name__ == '__main__':
    input = torch.randn(2, 32, 64, 64)
    m = Dynamic_conv2d(in_planes=32, out_planes=64, kernel_size=3, stride=2, padding=1, bias=False)
    out = m(input)
    print(out.shape)
