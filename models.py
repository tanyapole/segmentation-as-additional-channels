from torch import nn
from torchvision import models
from torch.optim import Adam


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
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
    elif args.model == 'resnext50':
            if args.pretrained:
                model = models.resnext50_32x4d(pretrained=True)
            else:
                model = models.resnext50_32x4d()
    elif args.model == 'resnet152':
        if args.pretrained:
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet152()
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
    if args.model in ['resnet50', 'resnext50', 'resnet152']:
        model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        last_layer_in_channels = model.fc.in_features
        model.fc = nn.Linear(last_layer_in_channels, out_shape)
    elif args.model == 'vgg16':
        model.features[0] = nn.Conv2d(input_num, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, out_shape)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    return model, optimizer
