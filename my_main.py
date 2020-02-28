import argparse
from pathlib import Path
import pandas as pd
import json
import os
from train import train
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnet152', 'inception_v3'])
    arg('--mask_use', action='store_true')
    arg('--root', type=str, default='runs/debug')
    arg('--N', type=int, default=1)
    arg('--batch-normalization', action='store_true')  # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float, nargs='*', default=[0.001])
    arg('--batch-size', type=int, default=1)
    arg('--augment-list', type=list, nargs='*', default=[])
    arg('--image-path', type=str, default='/home/irek/My_work/train/h5_224/')
    arg('--n-epochs', type=int, default=1)
    arg('--K-models', type=int, default=5)
    arg('--begin-number', type=int, default=20)
    arg('--show-model', action='store_true')
    arg('--jaccard-weight', type=float, default=0.)
    arg('--attribute', type=str, nargs='*', default='attribute_pigment_network')
    arg('--mode', type=str, default='simple', choices=['simple'])
    arg('--freezing', action='store_true')
    arg('--jac_train', action='store_true')
    arg('--cuda1', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    log = root.joinpath('train.log').open('at', encoding='utf8')

    results = pd.DataFrame(columns=['freeze_mode', 'lr', 'exp', 'train_mode', 'epoch', 'loss', 'prec',
                                    'recall'])
    N = args.N
    learning_rates = args.lr
    freeze_modes = [False, True]
    mask_use = [True, False]

    for m_use in mask_use:
        args.mask_use = m_use
        for mode in freeze_modes:
            args.freezing = mode
            for lr in learning_rates:
                args.lr = lr
                for experiment in range(N):
                    args.N = experiment
                    print('Использование масок на трейне {} Заморозка {}, шаг обучения {}, '
                          'номер эксперимента {}'.format(args.mask_use, args.freezing, args.lr, args.N))
                    if args.mode == 'simple':
                        i = 0
                        root.joinpath('params'+str(i)+'.json').write_text(
                            json.dumps(vars(args), indent=True, sort_keys=True))
                        results = train(args, results)
                        i += 1
                    elif args.jac_train:
                        configs = {'jaccard-weight': [0., 0.5, 1.]}
                        i = 0
                        for m in configs['jaccard-weight']:
                            args.jaccard_weight = m
                            root.joinpath('params' + str(i) + '.json').write_text(
                                json.dumps(vars(args), indent=True, sort_keys=True))
                            results = train(args, results)
                            i += 1
                    else:
                        print('strange')
    results.to_csv('Results/results.csv', index=False)
