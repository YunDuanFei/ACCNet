import torch
import math
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=32, norm_layer=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.proj_1 = nn.Sequential(
            nn.Conv2d(inplanes, width, 1),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        # light features extraction module
        self.lfem = nn.Sequential(
            nn.Conv2d(width, width, 5, stride=1, padding=2, groups=width),
            nn.Conv2d(width, width, 7, stride=stride, padding=9, groups=width, dilation=3),
            nn.Conv2d(width, width, 1),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.proj_2 = nn.Conv2d(width, planes * self.expansion, 1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.proj_1(x)
        out = self.lfem(out)
        out = self.proj_2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LAN(nn.Module):
    def __init__(self, block, layers, num_classes=6, iscap=None):
        super(LAN, self).__init__()
        self.inplanes = 64
        self._norm_layer = nn.BatchNorm2d
        self.iscap = iscap
        self.expansion = block.expansion
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.init_conv(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        if self.iscap:
            return self._forward_features(x)
        else:
            return self._forward_impl(x)

def _lan(arch, block, layers, **kwargs):
    model = LAN(block, layers, **kwargs)
    return model

def lan21(**kwargs):
    return _lan('lan', Bottleneck, [2, 2, 2, 2], **kwargs)

if __name__ == '__main__':
    model = lan21(num_classes=3)
    # print(model)
    x = torch.rand(1, 3, 72, 72)
    y = model(x)
    # print(y.shape)  # 1x128x18x18

















