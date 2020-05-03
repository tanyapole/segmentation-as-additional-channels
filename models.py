import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UnetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        self.up_transpose = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, inp):
        u, x = inp
        u = self.up_transpose(u)
        u = torch.cat((u, x), dim=1)
        u = self.conv_block_1(u)
        u = self.conv_block_2(u)
        return u


class ResUNet(nn.Module):

    def __init__(self, pretrained, classes):
        super().__init__()
        base_model = models.resnet50(pretrained=pretrained)

        self.down1 = nn.Sequential(*[base_model.conv1,
                                     base_model.bn1,
                                     base_model.relu])
        self.down2 = base_model.maxpool
        self.down3 = base_model.layer1
        self.down4 = base_model.layer2
        self.down5 = base_model.layer3
        self.down6 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, classes)

        self.MiddleBridge = nn.Sequential(*[ConvBlock(2048, 2048),
                                            ConvBlock(2048, 2048)])

        self.up1 = UnetUpBlock(2048, 1024)
        self.up2 = UnetUpBlock(1024, 512)
        self.up3 = UnetUpBlock(512, 256)
        self.up4 = UnetUpBlock(in_channels=128 + 64, out_channels=128,
                               up_conv_in_channels=256, up_conv_out_channels=128)
        self.up5 = UnetUpBlock(in_channels=64 + 3, out_channels=64,
                               up_conv_in_channels=128, up_conv_out_channels=64)

    def forward(self, x):
        x1 = self.down1(x)   # -> 112x112x64
        x2 = self.down2(x1)  # -> 56x56x64
        x3 = self.down3(x2)  # -> 56x56x256
        x4 = self.down4(x3)  # -> 28x28x512
        x5 = self.down5(x4)  # -> 14x14x1024
        x6 = self.down6(x5)  # -> 7x7x2048

        b = self.MiddleBridge(x6)

        z = self.up1((b, x5))  # -> 14x14x1024
        z = self.up2((z, x4))  # -> 28x28x512
        z = self.up3((z, x3))  # -> 56x56x256
        z = self.up4((z, x1))  # -> 112x112x128
        z = self.up5((z, x))   # -> 224x224x64

        x = self.avgpool(x6)   # classification
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, z


def create_model(args):

    model = ResUNet(args.pretrained, len(args.attribute))

    if args.freezing:
        for param in model.parameters():
            param.requires_grad = False

    return model
