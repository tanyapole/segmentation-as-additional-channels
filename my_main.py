import os
import argparse
import datetime

import pandas as pd

from train import train
from Utils.constants import SEED_LIST, TRAIN_TRAIN_NUMBER, TRAIN_VALID_NUMBER, PRETRAIN, YNET, BASELINE, K1, K2
from Utils.utils import print_save_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--N', type=int, default=1)
    arg('--pretrained', action='store_true')
    arg('--lr', type=float, default=0.0001)
    arg('--batch_size', type=int, default=1)
    arg('--image_path', type=str, default='./Data/h5/')
    arg('--attribute', nargs='+', default=[])
    arg('--cuda1', action='store_true')
    arg('--workers', type=int, default=1)
    arg('--resume', action='store_true')
    arg('--show_model', action='store_true')
    arg('--model_path', type=str, default='/Data/model/')
    arg('--normalize', action='store_true')
    args = parser.parse_args()

    results1 = pd.DataFrame()
    results2 = pd.DataFrame()
    results3 = pd.DataFrame()

    N = args.N

    time = datetime.datetime.now().strftime('%d_%H.%M')
    i = 0

    for experiment in range(0, N):
        SEED = SEED_LIST[experiment]
        args.N = experiment
        print(args)
        
        print('_' * 40)
        print('pretrain resnet50 on {} for {} epoch'.format(TRAIN_TRAIN_NUMBER, K1))
        results1 = train(args, results1, SEED=SEED, train_type=PRETRAIN, epochs=K1)
        print_save_results(args, results1, time, PRETRAIN)

        print('_' * 40)
        print('train y_net on {} for {} epoch'.format(TRAIN_TRAIN_NUMBER+TRAIN_VALID_NUMBER, K2))
        results2 = train(args, results2, SEED=SEED, train_type=YNET, epochs=K2)
        print_save_results(args, results2, time, YNET)

        """print('_' * 40)
        print('train resnet50 on {} for {} epoch to compare'.format(TRAIN_TRAIN_NUMBER+TRAIN_VALID_NUMBER, K1+K2))
        results3 = train(args, results3, SEED=SEED, train_type=BASELINE, epochs=K1+K2)
        print_save_results(args, results3, time, BASELINE)"""
