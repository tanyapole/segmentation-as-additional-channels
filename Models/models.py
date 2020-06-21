from torch import nn
from torchvision import models
from torch.optim import Adam
from torchvision.models import resnet50


def create_model(args):
    if args.model == 'vgg16':
        if args.batch_norm:
            model = models.vgg16_bn(pretrained=args.pretrained)
        else:
            model = models.vgg16(pretrained=args.pretrained)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)

    out_shape = len(args.attribute)

    if args.model == 'resnet50':
        last_layer_in_channels = model.fc.in_features
        model.fc = nn.Linear(last_layer_in_channels, out_shape)
    elif args.model == 'vgg16':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, out_shape)

    optimizer = Adam(model.parameters(), lr=args.lr)
    return model, optimizer
