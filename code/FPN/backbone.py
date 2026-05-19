import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.stage0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.stage1 = resnet.layer1  # C2 (256)
        self.stage2 = resnet.layer2  # C3 (512)
        self.stage3 = resnet.layer3  # C4 (1024)
        self.stage4 = resnet.layer4  # C5 (2048)

    def forward(self, x):
        x = self.stage0(x)
        C2 = self.stage1(x)
        C3 = self.stage2(C2)
        C4 = self.stage3(C3)
        C5 = self.stage4(C4)
        return C2, C3, C4, C5