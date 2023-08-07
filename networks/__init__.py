import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import resnet18
from .backbones.lan import lan21
from .backbones.convnext import convnext_tiny
from .backbones.efficientnet import efficientnet_b0
from .backbones.mobilenext import MobileNetXt
from .backbones.repvgg import RepVGG_A0
from .backbones.capsnet import PrimaryCapsLayer, AgreementRouting, CapsLayer

backbone_model_dict = {
    "resnet18": resnet18,
    'lan': lan21,
    'convnext': convnext_tiny,
    'efficientnet': efficientnet_b0,
    'mobilenext': MobileNetXt,
    'repvgg': RepVGG_A0,
}


class ResCapNet(nn.Module):
    def __init__(self, backbone, routing_iterations, n_classes, in_channels, hw, tcc, ecc, k, digit_dim, s):
        super(ResCapNet, self).__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.digit_dim = digit_dim
        assert in_channels == tcc*ecc, 'backbone output channels != capsule input channels'
        self.primaryCaps = PrimaryCapsLayer(in_channels, tcc, ecc, kernel_size=k, stride=s)  # outputs 5*5
        self.num_primaryCaps = tcc * int(math.floor((hw-k)/s+1)) * int(math.floor((hw-k)/s+1))
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, ecc, n_classes, digit_dim, routing_module)
        self.fc = nn.Linear(n_classes*digit_dim, n_classes*digit_dim)

    def forward(self, input):  # bx3x72x72
        b, c, h, w = input.shape
        x = self.backbone(input)  # bx128x18x18
        x = self.primaryCaps(x)  # 1x400x16
        x = self.digitCaps(x)  # 1x6x3
        x = self.fc(x.view(b, -1)).view(b, self.n_classes, self.digit_dim)
        return x

def create_net(args):
    # get backbone para
    backbone = None
    kwargs_backbone = {}
    if args.arch.lower() in ['resnet18', 'lan']:
        kwargs_backbone["iscap"] = args.iscap
        kwargs_backbone["num_classes"] = args.fcnum
    else:
        kwargs_backbone["num_classes"] = args.fcnum
        if args.iscap:
            raise NotImplementedError

    if args.iscap:
        backbone = backbone_model_dict[args.arch.lower()](**kwargs_backbone)
        # get capsule para
        kwargs_caps = {}
        kwargs_caps["routing_iterations"] = args.routing_iterations
        kwargs_caps["n_classes"] = args.capsules_num
        kwargs_caps["in_channels"] = args.backbone_channels
        kwargs_caps["hw"] = args.backbone_hw
        kwargs_caps["tcc"] = args.capsules_tcc
        kwargs_caps["ecc"] = args.capsules_ecc
        kwargs_caps["k"] = args.capsules_k
        kwargs_caps["digit_dim"] = args.digit_dim
        kwargs_caps["s"] = args.capsules_s
        rescapnet = ResCapNet(backbone, **kwargs_caps)
        return rescapnet
    else:
        backbone = backbone_model_dict[args.arch.lower()](**kwargs_backbone)
        return backbone
