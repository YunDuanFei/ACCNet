import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)  # 1x400
    lengths = lengths2.sqrt()  # 1x400
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), -1)  # 1x400x8
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))  # 400x3

    def forward(self, u_predict):  # 1x400x3x3
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=-1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)  # 广播后点积(逐元素相乘) 1x3x3
        v = squash(s)  # 1x10x16

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))  # 1x400x3
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)  # 1x1x3x3
                b_batch = b_batch + (u_predict * v).sum(-1)  # 1x400x3

                c = F.softmax(b_batch.view(-1, output_caps), dim=-1).view(-1, input_caps, output_caps, 1)  # 1x400x3x1
                s = (c * u_predict).sum(dim=1)  # 1x3x3
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))  # 400x8x18
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)  # 1x400x8
        u_predict = caps_output.matmul(self.weights)  # 高维矩阵乘法 1x400x1x18
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)  # 1x400x6x3
        v = self.routing_module(u_predict)  # 1x6x3
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)  # 1x128x5x5
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)  # 1x16x8x5x5

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # 1x16x5x5x8
        out = out.view(out.size(0), -1, out.size(4))  # 1x400x16
        out = squash(out)
        return out
