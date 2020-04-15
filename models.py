from torch import nn
from torchvision import models
from torch.optim import Adam


def create_model(args):

    if args.batch_norm:
        model = models.vgg16_bn(pretrained=args.pretrained)
    else:
        model = models.vgg16(pretrained=args.pretrained)

    if not args.freezing:
        for param in model.parameters():
            param.requires_grad = False

    out_shape = len(args.attribute)

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, out_shape)

    optimizer = Adam(model.parameters(), lr=args.lr)
    return model, optimizer
