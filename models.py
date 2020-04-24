from torch import nn
from torchvision import models
from torch.optim import Adam, SGD
from torchvision.models import resnext101_32x8d, resnet50
from torchvision.models.utils import load_state_dict_from_url
import torch

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


class VGG(nn.Module):

    def __init__(self, aux_layer_index, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.aux_layer_index = aux_layer_index
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        aux = self.features[:self.aux_layer_index](x)
        x = self.features[self.aux_layer_index:](aux)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, aux

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, aux_layer_index, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(aux_layer_index, make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, aux_layer_index=9, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, aux_layer_index, **kwargs)


def create_model(args):

    if args.model == 'vgg16':
        if args.batch_norm:
            model = models.vgg16_bn(pretrained=args.pretrained)
        else:
            # model = models.vgg16(pretrained=args.pretrained)
            model = vgg16(pretrained=args.pretrained, aux_layer_index=args.aux_layer_index)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.model =='resnext101':
        model = resnext101_32x8d(pretrained=args.pretrained)
    else:
        return

    if args.freezing:
        for param in model.parameters():
            param.requires_grad = False

    out_shape = len(args.attribute)

    # channels replacement
    if args.model in ['resnet50', 'resnext101']:
        if args.mask_use:
            input_num = 3 + len(args.attribute)
            model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        last_layer_in_channels = model.fc.in_features
        model.fc = nn.Linear(last_layer_in_channels, out_shape)
    elif args.model == 'vgg16':
        if args.mask_use:
            input_num = 3 + len(args.attribute)
            model.features[0] = nn.Conv2d(input_num, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(4096, out_shape)

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
    return model, optimizer
