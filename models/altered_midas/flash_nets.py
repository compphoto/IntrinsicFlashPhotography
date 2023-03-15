import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_scratch
)


def _calc_same_pad(i, k, s, d):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(
        x, weight, bias=None, stride=(1, 1),
        padding=(0, 0), dilation=(1, 1), groups=1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False, in_chan=3):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable
    )

    if in_chan != 3:
        efficientnet.conv_stem = Conv2dSame(in_chan, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


# there is a torch.nn.Flatten but idk I'm using this because
# I know the exact operation is does, so I won't mess things up
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class DecomposeNet(nn.Module):

    def __init__(self, activation='tanh', features=64, backbone="efficientnet_lite3", exportable=True, input_channels=3,
                 output_channels=1, align_corners=True, blocks={'expand': True}):

        super(DecomposeNet, self).__init__()

        use_pretrained = False

        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        if activation == 'sigmoid':
            output_act1 = nn.Sigmoid()
            output_act2 = nn.Sigmoid()
        if activation == 'tanh':
            output_act1 = nn.Tanh()
            output_act2 = nn.Tanh()
        if activation == 'none':
            output_act1 = nn.Identity()
            output_act2 = nn.Identity()

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, in_chan=input_channels,
                                                              exportable=exportable)
        self.scratch1 = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)
        self.scratch2 = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)

        # create the first decoder 
        self.scratch1.activation = nn.ReLU(False)

        self.scratch1.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch1.activation, deconv=False,
                                                             bn=False, align_corners=align_corners)

        self.scratch1.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch1.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act1
        )

        # create the second decoder
        self.scratch2.activation = nn.ReLU(False)

        self.scratch2.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch2.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch2.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch2.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch2.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch2.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch2.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch2.activation, deconv=False,
                                                             bn=False, align_corners=align_corners)

        self.scratch2.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch2.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act2
        )

        # create the ambient color decoder, this might be overkill, 
        # could try with just two linear layers, probably won't make a diff
        self.light_decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_sz = x.shape[0]

        # send the input through the encoder
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # send encoder output through the first decoder
        layer_1_rn_1 = self.scratch1.layer1_rn(layer_1)
        layer_2_rn_1 = self.scratch1.layer2_rn(layer_2)
        layer_3_rn_1 = self.scratch1.layer3_rn(layer_3)
        layer_4_rn_1 = self.scratch1.layer4_rn(layer_4)

        path_4_1 = self.scratch1.refinenet4(layer_4_rn_1)
        path_3_1 = self.scratch1.refinenet3(path_4_1, layer_3_rn_1)
        path_2_1 = self.scratch1.refinenet2(path_3_1, layer_2_rn_1)
        path_1_1 = self.scratch1.refinenet1(path_2_1, layer_1_rn_1)

        out_1 = self.scratch1.output_conv(path_1_1)

        # repeat the process for the second decoder
        layer_1_rn_2 = self.scratch2.layer1_rn(layer_1)
        layer_2_rn_2 = self.scratch2.layer2_rn(layer_2)
        layer_3_rn_2 = self.scratch2.layer3_rn(layer_3)
        layer_4_rn_2 = self.scratch2.layer4_rn(layer_4)

        path_4_2 = self.scratch2.refinenet4(layer_4_rn_2)
        path_3_2 = self.scratch2.refinenet3(path_4_2, layer_3_rn_2)
        path_2_2 = self.scratch2.refinenet2(path_3_2, layer_2_rn_2)
        path_1_2 = self.scratch2.refinenet1(path_2_2, layer_1_rn_2)

        out_2 = self.scratch2.output_conv(path_1_2)

        # finally get the lighting
        light = self.light_decoder(layer_4).view(batch_sz, 1)

        return out_1, out_2, light


class GenerateNet(nn.Module):

    def __init__(self, activation='tanh', features=64, backbone="efficientnet_lite3", exportable=True, input_channels=3,
                 output_channels=1, align_corners=True, blocks={'expand': True}):

        super(GenerateNet, self).__init__()

        use_pretrained = False

        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        if activation == 'sigmoid':
            output_act = nn.Sigmoid()
        if activation == 'tanh':
            output_act = nn.Tanh()
        if activation == 'none':
            output_act = nn.Identity()

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, in_chan=input_channels,
                                                              exportable=exportable)
        self.scratch1 = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)

        # create the first decoder 
        self.scratch1.activation = nn.ReLU(False)

        self.scratch1.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch1.activation, deconv=False,
                                                             bn=False, align_corners=align_corners)

        self.scratch1.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch1.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act
        )

    def forward(self, x):
        # send the input through the encoder
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # get the output using a single decoder
        layer_1_rn_1 = self.scratch1.layer1_rn(layer_1)
        layer_2_rn_1 = self.scratch1.layer2_rn(layer_2)
        layer_3_rn_1 = self.scratch1.layer3_rn(layer_3)
        layer_4_rn_1 = self.scratch1.layer4_rn(layer_4)

        path_4_1 = self.scratch1.refinenet4(layer_4_rn_1)
        path_3_1 = self.scratch1.refinenet3(path_4_1, layer_3_rn_1)
        path_2_1 = self.scratch1.refinenet2(path_3_1, layer_2_rn_1)
        path_1_1 = self.scratch1.refinenet1(path_2_1, layer_1_rn_1)

        out_1 = self.scratch1.output_conv(path_1_1)

        return out_1


class SimpleNet(nn.Module):

    def __init__(self, activation='tanh', features=64, backbone="efficientnet_lite3", exportable=True, input_channels=3,
                 output_channels=3, align_corners=True, blocks={'expand': True}):

        super(SimpleNet, self).__init__()

        use_pretrained = False

        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        if activation == 'sigmoid':
            output_act = nn.Sigmoid()
        if activation == 'tanh':
            output_act = nn.Tanh()
        if activation == 'none':
            output_act = nn.Identity()

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, in_chan=input_channels,
                                                              exportable=exportable)
        self.scratch1 = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)

        # create the first decoder
        self.scratch1.activation = nn.ReLU(False)

        self.scratch1.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch1.activation, deconv=False,
                                                             bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch1.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch1.activation, deconv=False,
                                                             bn=False, align_corners=align_corners)

        self.scratch1.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch1.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act
        )

    def forward(self, x):
        # send the input through the encoder
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # get the output using a single decoder
        layer_1_rn_1 = self.scratch1.layer1_rn(layer_1)
        layer_2_rn_1 = self.scratch1.layer2_rn(layer_2)
        layer_3_rn_1 = self.scratch1.layer3_rn(layer_3)
        layer_4_rn_1 = self.scratch1.layer4_rn(layer_4)

        path_4_1 = self.scratch1.refinenet4(layer_4_rn_1)
        path_3_1 = self.scratch1.refinenet3(path_4_1, layer_3_rn_1)
        path_2_1 = self.scratch1.refinenet2(path_3_1, layer_2_rn_1)
        path_1_1 = self.scratch1.refinenet1(path_2_1, layer_1_rn_1)

        out_1 = self.scratch1.output_conv(path_1_1)

        return out_1
