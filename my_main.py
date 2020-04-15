import argparse
from pathlib import Path
import pandas as pd
import os
from train import train
import datetime
from Utils.utils import print_save_results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnext101', 'resnext50'])
    arg('--mask_use', action='store_true')
    arg('--root', type=str, default='runs/debug')
    arg('--N', type=int, default=1)
    arg('--batch_norm', action='store_true')           # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float, nargs='*', default=[0.0001])
    arg('--batch_size', type=int, default=1)
    arg('--augment_list', nargs='+', default=[])
    arg('--image_path', type=str, default='./Data/h5/')
    arg('--n_epochs', type=int, default=1)
    arg('--show_model', action='store_true')
    arg('--model_path', type=str, default='/Data/model/')
    arg('--prob', type=float, nargs='*', default=[0.1])
    arg('--attribute', nargs='+', default=[])
    arg('--freezing', action='store_true')
    arg('--cuda1', action='store_true')
    arg('--cell', action='store_true')
    arg('--cell_size', type=int, nargs='*', default=[56])
    arg('--workers', type=int, default=1)
    arg('--resume', action='store_true')
    arg('--aux', action='store_true')
    arg('--aux_batch', type=int, default=4)
    arg('--optimizer', type=str, choices=['adam', 'sgd'])
    arg('--save_model', action='store_true')
    arg('--pos_weight', action='store_true')
    arg('--weights', type=float, nargs='*', default=[])
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    log = root.joinpath('train.log').open('at', encoding='utf8')

    results = pd.DataFrame()

    N = args.N
    learning_rates = args.lr
    cell = [False]
    cell_size = args.cell_size
    probs = args.prob
    time = datetime.datetime.now().strftime('%d_%H.%M')
    i = 0
    r = random.Random(0)
    SEED_LIST = [r.randint(1, 500) for _ in range(10)]
    best_f1 = 0

    if args.aux:
        for lr in learning_rates:
            args.lr = lr
            for experiment in range(N):
                args.N = experiment
                print(args)
                results, best_f1 = train(args, results, best_f1)
                print_save_results(args, results, root, i, time)
                i += 1
    else:
        for lr in learning_rates:
            args.lr = lr
            if args.mask_use:
                for c in cell:
                    args.cell = c
                    if args.cell:
                        for cs in cell_size:
                            args.cell_size = cs
                            for p in probs:
                                args.prob = p
                                for experiment in range(N):
                                    args.N = experiment
                                    print(args)
                                    results, best_f1 = train(args, results, best_f1, seed=SEED_LIST[experiment])
                                    print_save_results(args, results, root, i, time)
                                    i += 1
                    else:
                        for p in probs:
                            args.prob = p
                            for experiment in range(N):
                                print(args)
                                args.N = experiment
                                results, best_f1 = train(args, results, best_f1, seed=SEED_LIST[experiment])
                                print_save_results(args, results, root, i, time)
                                i += 1
            else:
                for experiment in range(N):
                    args.N = experiment
                    print(args)
                    results, best_f1 = train(args, results, best_f1, seed=SEED_LIST[experiment])
                    print_save_results(args, results, root, i, time)
                    i += 1
