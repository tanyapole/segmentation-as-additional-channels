import torch
from torch import nn
from torchvision import models

from Utils.constants import YNET, PRETRAIN


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


class ResYNet(nn.Module):

    def __init__(self, pretrained: bool, n_class: int, model_path : str='', N: int=0):
        super().__init__()

        base_model = models.resnet50(pretrained=pretrained)
        base_model.fc = nn.Linear(2048, n_class)

        checkpoint = torch.load(model_path + 'model_{}.pt'.format(N))
        base_model.load_state_dict(checkpoint['model'])

        self.down1 = nn.Sequential(*[base_model.conv1,
                                     base_model.bn1,
                                     base_model.relu])
        self.down2 = base_model.maxpool
        self.down3 = base_model.layer1
        self.down4 = base_model.layer2
        self.down5 = base_model.layer3
        self.down6 = base_model.layer4

        self.clsf = nn.Sequential(*[ConvBlock(2048, 256, kernel_size=1, padding=0),
                                    base_model.avgpool,
                                    nn.Flatten(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, n_class, bias=False)])

        self.MiddleBridge = nn.Sequential(*[ConvBlock(2048, 2048),
                                            ConvBlock(2048, 2048)])
        # self.Bridge1 = ConvBlock(2048, 2048)
        # self.Bridge2 = ConvBlock(2048, 2048)

        self.up1 = UnetUpBlock(2048, 1024)
        self.up2 = UnetUpBlock(1024, 512)
        self.up3 = UnetUpBlock(512, 256)
        self.up4 = UnetUpBlock(in_channels=128 + 64, out_channels=128,
                               up_conv_in_channels=256, up_conv_out_channels=128)
        self.up5 = UnetUpBlock(in_channels=64 + 3, out_channels=64,
                               up_conv_in_channels=128, up_conv_out_channels=64)
        self.conv_segm = nn.Sequential(*[ConvBlock(64, 16, kernel_size=1, padding=0),
                                         nn.Conv2d(16, n_class, 1)])

    def forward(self, x):
        x1 = self.down1(x)   # -> 112x112x64
        x2 = self.down2(x1)  # -> 56x56x64
        x3 = self.down3(x2)  # -> 56x56x256
        x4 = self.down4(x3)  # -> 28x28x512
        x5 = self.down5(x4)  # -> 14x14x1024
        x6 = self.down6(x5)  # -> 7x7x2048

        b = self.MiddleBridge(x6)
        # b = self.Bridge1(x6)
        # z = self.Bridge2(b)

        z = self.up1((b, x5))  # -> 14x14x1024
        z = self.up2((z, x4))  # -> 28x28x512
        z = self.up3((z, x3))  # -> 56x56x256
        z = self.up4((z, x1))  # -> 112x112x128
        z = self.up5((z, x))   # -> 224x224x64
        z = self.conv_segm(z)  # -> 224x224xn

        x = self.clsf(x6)   # classification

        return x, z


class SResYNet(nn.Module):

    def __init__(self, pretrained: bool, n_class: int, model_path : str='', N: int=0):
        super().__init__()

        base_model = Unet(pretrained, n_class)
        checkpoint = torch.load(model_path + 'model_{}.pt'.format(N))
        base_model.load_state_dict(checkpoint['model'])

        self.down1 = base_model.down1
        self.down2 = base_model.down2
        self.down3 = base_model.down3
        self.down4 = base_model.down4
        self.down5 = base_model.down5
        self.down6 = base_model.down6

        self.clsf = nn.Sequential(*[ConvBlock(512, 256, kernel_size=1, padding=0),
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, n_class, bias=False)])

        self.MiddleBridge = base_model.MiddleBridge

        self.up1 = base_model.up1
        self.up2 = base_model.up2
        self.up3 = base_model.up3
        self.up4 = base_model.up4
        self.up5 = base_model.up5
        self.conv_segm = base_model.conv_segm

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
        z = self.conv_segm(z)  # -> 224x224xn

        x = self.clsf(x4)

        return x, z


class Unet(nn.Module):

    def __init__(self, pretrained: bool, n_class: int):
        super().__init__()

        base_model = models.resnet50(pretrained=pretrained)
        base_model.fc = nn.Linear(2048, n_class)

        self.down1 = nn.Sequential(*[base_model.conv1,
                                     base_model.bn1,
                                     base_model.relu])
        self.down2 = base_model.maxpool
        self.down3 = base_model.layer1
        self.down4 = base_model.layer2
        self.down5 = base_model.layer3
        self.down6 = base_model.layer4

        self.MiddleBridge = nn.Sequential(*[ConvBlock(2048, 2048),
                                            ConvBlock(2048, 2048)])

        self.up1 = UnetUpBlock(2048, 1024)
        self.up2 = UnetUpBlock(1024, 512)
        self.up3 = UnetUpBlock(512, 256)
        self.up4 = UnetUpBlock(in_channels=128 + 64, out_channels=128,
                               up_conv_in_channels=256, up_conv_out_channels=128)
        self.up5 = UnetUpBlock(in_channels=64 + 3, out_channels=64,
                               up_conv_in_channels=128, up_conv_out_channels=64)
        self.conv_segm = nn.Sequential(*[ConvBlock(64, 16, kernel_size=1, padding=0),
                                         nn.Conv2d(16, n_class, 1)])

    def forward(self, x):

        x1 = self.down1(x)  # -> 112x112x64

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
        z = self.up5((z, x))  # -> 224x224x64
        z = self.conv_segm(z)  # -> 224x224xn

        return z


def create_model(args, train_type):
    if train_type == YNET:
        model = SResYNet(args.pretrained, len(args.attribute), args.model_path, args.N)
    elif train_type == PRETRAIN:
        model = Unet(args.pretrained, len(args.attribute))
    else:
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(2048, len(args.attribute))
    return model
