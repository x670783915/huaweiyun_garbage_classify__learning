import torch
import torch.nn as nn

class SE_module(nn.Module):

    def __init__(self, channel, r):
        super(SE_module, self).__init__()
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, bias=False),  # r 是为了减少计算成本减少了MLP隐含层的参数个数,
                                                              # 同时也增强了非线性能力
            nn.ReLU(True),
            nn.Conv2d(channel // r, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.__avg_pool(x)
        y = self.__fc(y)
        return x * y

class Channel_Attention(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel // r , 1, bias=False),
            nn.ReLU(True), #inplace = True 代表原地执行方法 类似 max_ etc
            nn.Conv2d(channel // r, channel, 1, bias=False),
        )

        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__avg_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1 + y2)

        return y

class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1

        padding = (kernel_size - 1) // 2
        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask
