import os
import random
import argparse
import datetime

import pandas as pd
from pathlib import Path

from train import train
from Utils.utils import print_save_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--root', type=str, default='runs/debug')
    arg('--N', type=int, default=1)
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float, default=0.0001)
    arg('--batch_size', type=int, default=1)
    arg('--augment_list', nargs='+', default=[])
    arg('--image_path', type=str, default='./Data/h5/')
    arg('--n_epochs', type=int, default=1)
    arg('--prob', type=float, nargs='*', default=[0.1])
    arg('--attribute', nargs='+', default=[])
    arg('--cuda1', action='store_true')
    arg('--workers', type=int, default=1)
    arg('--resume', action='store_true')
    arg('--show_model', action='store_true')
    arg('--save_model', action='store_true')
    arg('--model_path', type=str, default='/Data/model/')
    arg('--normalize', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    results = pd.DataFrame()

    N = args.N

    time = datetime.datetime.now().strftime('%d_%H.%M')
    i = 0
    r = random.Random(0)
    SEED_LIST = [r.randint(1, 500) for _ in range(10)]

    for experiment in range(1, N):
        args.N = experiment
        print(args)
        results = train(args, results, SEED=SEED_LIST[experiment])
        print_save_results(args, results, root, i, time)
        i += 1
