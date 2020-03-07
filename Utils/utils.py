import json

import h5py
import torch
import pandas as pd
import os
import sys


def load_image(file_name, type='image'):
    f = h5py.File(file_name, 'r')
    file_np = f['img'][()]
    if type == 'image':
        file_np = (file_np / 255.0).astype('float32')
    elif type == 'mask':
        file_np = file_np.astype('uint8')
    else:
        print('not choosed type to load')
        return
    return file_np


def print_save_results(args, results, root, i, time):
    print('Использование масок на трейне {} Заморозка {}, шаг обучения {}, '
          'номер эксперимента {}'.format(args.mask_use, args.freezing, args.lr, args.N))
    root.joinpath('params {}_num {} .json'.format(time, i)).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    path = 'Results/{}'.format(time)
    name = 'results.csv'
    file = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    results.to_csv(file, index=False)


def write_tensorboard(writer, metrics, args):

    train_metrics, valid_metrics = metrics
    writer.add_scalars('loss', {'train/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): train_metrics['loss'],
                                'valid/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): valid_metrics['loss']},
                       train_metrics['epoch'])

    writer.add_scalars('precision', {'train/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): train_metrics['precision'],
                                     'valid/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): valid_metrics['precision']},
                       train_metrics['epoch'])

    writer.add_scalars('recall', {'train/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): train_metrics['recall'],
                                  'valid/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): valid_metrics['recall']},
                       train_metrics['epoch'])

    writer.add_scalars('accuracy', {'train/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): train_metrics['accuracy'],
                                    'valid/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                        args.N, args.lr): valid_metrics['accuracy']},
                       train_metrics['epoch'])

    writer.add_scalars('f1 score', {'train/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                                                                     args.N, args.lr): train_metrics[
        'f1_score'],
                                    'valid/mask{}/freeze{}/experiment{}/lr{}'.format(args.mask_use, args.freezing,
                                                                                     args.N, args.lr): valid_metrics[
                                        'f1_score']},
                       train_metrics['epoch'])


def save_weights(model, model_path, ep, train_metrics, valid_metrics):
    torch.save({'model': model.state_dict(),
                'epoch_time': ep,
                'valid_loss': valid_metrics['loss1'],
                'train_loss': train_metrics['loss1']},
               str(model_path)
               )
