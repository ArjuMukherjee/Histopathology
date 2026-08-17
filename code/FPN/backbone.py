import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    densenet121,
    DenseNet121_Weights,
)


def _get_resnet_channels(resnet):
    def _layer_channels(layer):
        block = layer[-1]
        if hasattr(block, 'bn3'):
            return block.bn3.num_features
        return block.bn2.num_features

    return [
        _layer_channels(resnet.layer1),
        _layer_channels(resnet.layer2),
        _layer_channels(resnet.layer3),
        _layer_channels(resnet.layer4),
    ]


class ResNetBackbone(nn.Module):
    def __init__(self, variant="resnet50", pretrained=False):
        super().__init__()

        weights_map = {
            "resnet18": ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        }

        if variant not in weights_map:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        resnet_fn = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }[variant]

        weights = weights_map[variant] if pretrained else None
        resnet = resnet_fn(weights=weights)

        self.stage0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.stage1 = resnet.layer1
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4

        self.out_channels = _get_resnet_channels(resnet)

    def forward(self, x):
        x = self.stage0(x)
        C2 = self.stage1(x)
        C3 = self.stage2(C2)
        C4 = self.stage3(C3)
        C5 = self.stage4(C4)
        return C2, C3, C4, C5


class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v2(weights=weights)

        self.stage0 = nn.Sequential(
            model.features[0],
            model.features[1],
            model.features[2],
        )
        self.stage1 = nn.Sequential(*model.features[3:7])
        self.stage2 = nn.Sequential(*model.features[7:14])
        self.stage3 = nn.Sequential(*model.features[14:])

        self.out_channels = [24, 32, 96, 1280]

    def forward(self, x):
        C2 = self.stage0(x)
        C3 = self.stage1(C2)
        C4 = self.stage2(C3)
        C5 = self.stage3(C4)
        return C2, C3, C4, C5


class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)

        self.stage0 = nn.Sequential(
            model.features[0],
            model.features[1],
            model.features[2],
        )
        self.stage1 = nn.Sequential(model.features[3])
        self.stage2 = nn.Sequential(model.features[4])
        self.stage3 = nn.Sequential(*model.features[5:])

        self.out_channels = [24, 40, 80, 1280]

    def forward(self, x):
        C2 = self.stage0(x)
        C3 = self.stage1(C2)
        C4 = self.stage2(C3)
        C5 = self.stage3(C4)
        return C2, C3, C4, C5


class DenseNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)

        self.stage0 = nn.Sequential(*model.features[:5])
        self.stage1 = nn.Sequential(model.features[5])
        self.stage2 = nn.Sequential(model.features[6], model.features[7])
        self.stage3 = nn.Sequential(*model.features[8:])

        self.out_channels = [256, 128, 256, 1024]

    def forward(self, x):
        C2 = self.stage0(x)
        C3 = self.stage1(C2)
        C4 = self.stage2(C3)
        C5 = self.stage3(C4)
        return C2, C3, C4, C5


def build_backbone(backbone_name="resnet50", pretrained=False):
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet20":
        backbone_name = "resnet18"

    if backbone_name in {"resnet18", "resnet34", "resnet50"}:
        return ResNetBackbone(variant=backbone_name, pretrained=pretrained)
    if backbone_name == "mobilenet_v2":
        return MobileNetV2Backbone(pretrained=pretrained)
    if backbone_name == "efficientnet_b0":
        return EfficientNetBackbone(pretrained=pretrained)
    if backbone_name == "densenet121":
        return DenseNetBackbone(pretrained=pretrained)

    raise ValueError(
        f"Unsupported backbone: {backbone_name}. Supported: resnet18, resnet34, resnet50, mobilenet_v2, efficientnet_b0, densenet121"
    )