import os
import random
import argparse
import datetime

from pathlib import Path
import pandas as pd

from Training.train import train
from Utils.utils import print_save_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnext101', 'resnext50'])
    arg('--mask_use', action='store_true')
    arg('--root', type=str, default='runs/debug')
    arg('--N', type=int, default=1)
    arg('--batch_norm', action='store_true')           # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float)
    arg('--batch_size', type=int, default=1)
    arg('--image_path', type=str, default='./Data/h5/')
    arg('--n_epochs', type=int, default=1)
    arg('--show_model', action='store_true')
    arg('--save_model', action='store_true')
    arg('--model_path', type=str, default='/data/ISIC/models/')
    arg('--attribute', nargs='+', default=[])
    arg('--cuda1', action='store_true')
    arg('--workers', type=int, default=1)
    arg('--optimizer', type=str, choices=['adam', 'sgd'])
    arg('--normalize', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    log = root.joinpath('train.log').open('at', encoding='utf8')

    results = pd.DataFrame()
    N = args.N
    time = datetime.datetime.now().strftime('%d_%H.%M')
    i = 0
    r = random.Random(0)
    SEED_LIST = [r.randint(1, 500) for _ in range(10)]

    for experiment in range(N):
        args.N = experiment
        print(args)
        results = train(args, results, SEED=SEED_LIST[experiment])
        print_save_results(args, results, root, i, time)
        i += 1
