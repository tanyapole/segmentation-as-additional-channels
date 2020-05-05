import json
import os
import random

import torch
import pandas as pd
import numpy as np

from Utils.constants import TRAIN_TRAIN_NUMBER, TRAIN_VALID_NUMBER, PRETRAIN


def read_split_data(SEED: int, train_type: str) -> pd.DataFrame:
    train_test_id = pd.read_csv('Data/train_test_id_with_masks.csv')
    indexes = np.arange(train_test_id.shape[0])
    random.seed(SEED)
    random.shuffle(indexes)
    train_test_id = train_test_id.iloc[indexes].reset_index(drop=True)
    train_test_id.loc[:, 'Split'] = ''
    if train_type == PRETRAIN:
        train_test_id.loc[:TRAIN_TRAIN_NUMBER, 'Split'] = 'train'
        train_test_id.loc[TRAIN_TRAIN_NUMBER:TRAIN_TRAIN_NUMBER + TRAIN_VALID_NUMBER, 'Split'] = 'valid'
        print(TRAIN_TRAIN_NUMBER, TRAIN_TRAIN_NUMBER + TRAIN_VALID_NUMBER)
    else:
        train_test_id.loc[:TRAIN_TRAIN_NUMBER+TRAIN_VALID_NUMBER, 'Split'] = 'train'
        train_test_id.loc[TRAIN_TRAIN_NUMBER+TRAIN_VALID_NUMBER:, 'Split'] = 'valid'
    return train_test_id


def print_save_results(args, results: pd.DataFrame, time: str, postfix: str):
    print('номер эксперимента {}'.format(args.N))
    path = 'Results/{}'.format(time)
    name = 'results_{}.csv'.format(postfix)
    file = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    results.to_csv(file, index=False)


def print_update(metrics, results: pd.DataFrame, args, mode: str, train_type: str) -> pd.DataFrame:
    print('''Epoch: {} Loss: {:.6f} bce1_loss {:.6f} train_type {} F1: {:.4f} F1_labeled {} 
             Time: {:.4f}'''.format(metrics['epoch'],
                                    metrics['loss'],
                                    metrics['bce1_loss'],
                                    train_type,
                                    metrics['f1_score'],
                                    metrics['f1_score_labels'],
                                    metrics['epoch_time']))

    results = results.append({'train_type': train_type,
                              'lr': args.lr,
                              'bce1_loss': metrics['bce1_loss'],
                              'bce2_loss': metrics['bce2_loss'],
                              'exp': args.N,
                              'train_mode': mode,
                              'epoch': metrics['epoch'],
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
