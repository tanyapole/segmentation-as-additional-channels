import os
import random
import argparse
import datetime

import pandas as pd
from pathlib import Path

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
    arg('--lr', type=float, default=0.0001)
    arg('--batch_size', type=int, default=1)
    arg('--augment_list', nargs='+', default=[])
    arg('--image_path', type=str, default='./Data/h5/')
    arg('--n_epochs', type=int, default=1)
    arg('--prob', type=float, nargs='*', default=[0.1])
    arg('--attribute', nargs='+', default=[])
    arg('--freezing', action='store_true')
    arg('--selective_freeze', action='store_true')
    arg('--cuda1', action='store_true')
    arg('--cell', action='store_true')
    arg('--cell_size', type=int, nargs='*', default=[56])
    arg('--workers', type=int, default=1)
    arg('--resume', action='store_true')
    arg('--optimizer', type=str, choices=['adam', 'sgd'])
    arg('--show_model', action='store_true')
    arg('--save_model', action='store_true')
    arg('--model_path', type=str, default='/Data/model/')
    arg('--normalize', action='store_true')
    arg('--pos_weight', action='store_true')
    arg('--weights', type=float, nargs='*', default=[])
    arg('--two_gpu', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    results = pd.DataFrame()

    N = args.N

    if args.mask_use:
        cell = [False]
        cell_size = args.cell_size
        probs = args.prob

    time = datetime.datetime.now().strftime('%d_%H.%M')
    i = 0
    r = random.Random(0)
    SEED_LIST = [r.randint(1, 500) for _ in range(10)]

    for experiment in range(2,2+N):
        args.N = experiment
        if args.mask_use:  # zeroing branch
            for c in cell:
                args.cell = c
                if args.cell:
                    for cs in cell_size:
                        args.cell_size = cs
                        for p in probs:
                            args.prob = p
                            print(args)
                            results = train(args, results, SEED=SEED_LIST[experiment])
                            print_save_results(args, results, root, i, time)
                            i += 1
                else:
                    for p in probs:
                        args.prob = p
                        print(args)
                        results = train(args, results, SEED=SEED_LIST[experiment])
                        print_save_results(args, results, root, i, time)
                        i += 1
        else:
            print(args)  # base train branch
            results = train(args, results, SEED=SEED_LIST[experiment])
            print_save_results(args, results, root, i, time)
            i += 1