import json
import os
import random

import torch
import pandas as pd
import numpy as np


def read_split_data(SEED: int, border: int = 1800) -> pd.DataFrame:
    train_test_id = pd.read_csv('Data/train_test_id_with_masks.csv')
    indexes = np.arange(train_test_id.shape[0])
    random.seed(SEED)
    random.shuffle(indexes)
    train_test_id = train_test_id.iloc[indexes].reset_index(drop=True)
    train_test_id.loc[:border, 'Split'] = 'train'
    train_test_id.loc[border:, 'Split'] = 'valid'
    return train_test_id


def print_save_results(args, results, root, i, time):
    print('Модель {}, mask usage {}, заморозка {}, lr {}, номер эксперимента {} размер квадрата {}, вероятность {}'
          .format(args.model, args.mask_use, args.freezing, args.lr, args.N, args.cell_size, args.prob))
    root.joinpath('params_{}_num {} .json'.format(time, i)).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    path = 'Results/{}'.format(time)
    name = 'results.csv'
    file = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    results.to_csv(file, index=False)


def print_update(metrics, results: pd.DataFrame, args, mode: str) -> pd.DataFrame:
    print('''Epoch: {} Loss: {:.6f} Pair_loss{:.6f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} F1_labeled {} 
             Time: {:.4f}'''.format(metrics['epoch'],
                                    metrics['loss'],
                                    metrics['pair_loss'],
                                    metrics['accuracy'],
                                    metrics['precision'],
                                    metrics['recall'],
                                    metrics['f1_score'],
                                    metrics['f1_score_labels'],
                                    metrics['epoch_time']))

    results = results.append({'model': args.model,
                              'lr': args.lr,
                              'pair': args.aux,
                              'bce_loss': metrics['bce_loss'],
                              'pair_loss': metrics['pair_loss'],
                              'exp': args.N,
                              'train_mode': mode,
                              'epoch': metrics['epoch'],
                              'cell': args.cell,
                              'cell_size': args.cell_size,
                              'prob': args.prob,
                              'loss': metrics['loss'],
                              'acc': metrics['accuracy'],
                              'acc_labels': metrics['accuracy_labels'],
                              'prec': metrics['precision'],
                              'prec_labels': metrics['precision_labels'],
                              'recall': metrics['recall'],
                              'recall_labels': metrics['recall_labels'],
                              'f1': metrics['f1_score'],
                              'f1_labels': metrics['f1_score_labels']}, ignore_index=True)

    return results


def channels_first(arr:np.array, channel:int=0) -> np.array: return np.moveaxis(arr, -1, channel)
def npy_to_float_tensor(arr:np.array) -> torch.Tensor: return torch.tensor(arr, dtype=torch.float32)


def save_weights(model, model_path, epoch, optimizer):
    torch.save({'model': model.module.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()},
               str(model_path)
               )
