import argparse
from pathlib import Path
import pandas as pd
import os
from train import train
import datetime
from Utils.utils import print_save_results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnet152', 'inception_v3'])
    arg('--mask_use', action='store_true')
    arg('--root', type=str, default='runs/debug')
    arg('--N', type=int, default=1)
    arg('--batch_normalization', action='store_true')  # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float, nargs='*', default=[0.001])
    arg('--batch_size', type=int, default=1)
    arg('--augment_list', type=list, nargs='*', default=[])
    arg('--image_path', type=str, default='./Data/h5/')
    arg('--n_epochs', type=int, default=1)
    arg('--show_model', action='store_true')
    arg('--prob', type=float, nargs='*', default=0.1)
    arg('--jaccard_weight', type=float, default=0.5)
    arg('--attribute', type=str, nargs='*', default='attribute_pigment_network')
    arg('--freezing', action='store_true')
    arg('--jac_train', action='store_true')
    arg('--cuda1', action='store_true')
    arg('--cell', action='store_true')
    arg('--cell_size', type=int, nargs='*', default=[56])
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    log = root.joinpath('train.log').open('at', encoding='utf8')

    results = pd.DataFrame(columns=['mask_use', 'freeze_mode', 'lr', 'exp', 'cell_size', 'prob', 'train_mode', 'epoch',
                                    'loss', 'prec', 'recall'])
    N = args.N
    learning_rates = args.lr
    freeze_modes = [False]
    mask_use = [True, False]
    cell_size = args.cell_size
    probs = args.prob
    time = datetime.datetime.now().strftime('%d %H:%M')
    i = 0

    for m_use in mask_use:
        args.mask_use = m_use
        for mode in freeze_modes:
            args.freezing = mode
            for lr in learning_rates:
                args.lr = lr
                for cs in cell_size:
                    args.cell_size = cs
                    for p in probs:
                        args.prob = p
                        for experiment in range(N):
                            args.N = experiment
                            results = train(args, results)
                            print_save_results(args, results, root, i, time)
                            i += 1
