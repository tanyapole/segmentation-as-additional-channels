from torch import nn
from torchvision import models
from torch.optim import Adam, SGD
from torchvision.models import resnext101_32x8d, resnet50


def create_model(args):

    if args.model == 'vgg16':
        if args.batch_norm:
            model = models.vgg16_bn(pretrained=args.pretrained)
        else:
            model = models.vgg16(pretrained=args.pretrained)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.model =='resnext101':
        model = resnext101_32x8d(pretrained=args.pretrained)
    else:
        return

    if not args.freezing:
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
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, out_shape)

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
    return model, optimizer
