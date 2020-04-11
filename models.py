from torch import nn
from torchvision import models
from torch.optim import Adam, SGD
from torchvision.models.resnet import model_urls, conv1x1, conv3x3
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y = self.layer2(x)
        x = self.layer3(y)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, y

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



def create_model(args, device):

    if args.model == 'vgg16':
        if args.pretrained:
            if args.batch_normalization:
                model = models.vgg16_bn(pretrained=True)
            else:
                model = models.vgg16(pretrained=True)
        else:
            if args.batch_normalization:
                model = models.vgg16_bn()
            else:
                model = models.vgg16()
    elif args.model == 'resnet50':
        if args.pretrained:
            model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=True, progress=True)
        else:
            model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=False, progress=True)
    else:
        return

    if not args.freezing:
        for param in model.parameters():
            param.requires_grad = False

    if args.mask_use:
        input_num = 3 + len(args.attribute)
        out_shape = len(args.attribute)
    else:
        input_num = 3
        out_shape = len(args.attribute)

    # channels replacement
    if args.model in ['resnet50', 'resnet152']:
        model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        last_layer_in_channels = model.fc.in_features
        model.fc = nn.Linear(last_layer_in_channels, out_shape)
    elif args.model == 'vgg16':
        model.features[0] = nn.Conv2d(input_num, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, out_shape)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    return model, optimizer
