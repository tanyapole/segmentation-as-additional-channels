import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn
from pathlib import Path
from my_dataset import make_loader
from models import create_model
from metrics import Metrics
from Utils.utils import save_weights
import random
from torchvision import transforms


def train(args, results, seed):

    epoch = 0

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = pd.read_csv('Data/train_test_id_with_masks.csv')
    indexes = np.arange(train_test_id.shape[0])
    random.seed(seed)
    random.shuffle(indexes)
    train_test_id = train_test_id.iloc[indexes].reset_index(drop=True)
    #train_test_id = train_test_id.sample(frac=1).reset_index(drop=True)
    train_test_id.loc[:1800, 'Split'] = 'train'
    train_test_id.loc[1800:, 'Split'] = 'valid'

    # uncomment for debugging
    """train_loader = make_loader(train_test_id, args, annotated, train='train', shuffle=True)
    print('--' * 10)
    print('check data')
    train_image, train_labels_ind, name = next(iter(train_loader))
    print('train_image.shape', train_image.shape)
    print('train_label_ind.shape', train_labels_ind.shape)
    print('train_image.min', train_image.min().item())
    print('train_image.max', train_image.max().item())
    print('train_label_ind.min', train_labels_ind.min().item())
    print('train_label_ind.max', train_labels_ind.max().item())
    print('--' * 10)"""

    cudnn.benchmark = True

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    model, optimizer = create_model(args)

    if args.show_model:
        print(model)

    model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    metric = Metrics(args)
    best_f1 = 10 ** 8
    for ep in range(epoch, args.n_epochs):
        try:
            metrics = [0, 0]
            for i, mode in enumerate(['train', 'valid']):
                metrics[i], results = make_step(model=model, mode=mode, train_test_id=train_test_id,
                                                args=args, device=device, criterion=criterion,
                                                optimizer=optimizer, results=results, metric=metric, epoch=ep)

            if args.save_model and ep < 100:
                if metrics[0]['loss'] < best_f1:
                    name = '{}model_{}.pt'.format(args.model_path, args.N)
                    save_weights(model, name, metrics, optimizer)
                    best_f1 = metrics[0]['loss']

        except KeyboardInterrupt:
            return
    return results


def make_step(model, mode, train_test_id, args, device, criterion, optimizer, results, metric, epoch):
    start_time = time.time()
    if mode == 'train':
        model.train()
    else:
        model.eval()
    loader = make_loader(train_test_id, args, train=mode, shuffle=True)
    n = len(loader)
    if mode == 'valid':
        torch.set_grad_enabled(False)
        model.eval()
    for i, (image_batch, labels_batch, names) in enumerate(loader):
        if i == n - 1:
            print(f'\r', end='')
        elif i < n - 3:
            print(f'\rBatch {i} / {n} ', end='')
        # print(np.unique(image_batch))
        image_batch = image_batch.to(device).type(torch.cuda.FloatTensor)
        labels_batch = labels_batch.to(device).type(torch.cuda.FloatTensor)

        last_output = model(image_batch)

        if isinstance(args.attribute, str):
            labels_batch = torch.reshape(labels_batch, (-1, 1))

        outputs = nn.Sigmoid()(last_output)
        #loss = criterion(last_output, labels_batch)
        loss = nn.BCELoss()(outputs, labels_batch)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        outputs = np.around(outputs.data.cpu().numpy())
        labels_batch = labels_batch.data.cpu().numpy()
        #print(outputs, labels_batch)
        metric.update(labels_batch, outputs, loss)

    epoch_time = time.time() - start_time
    metrics = metric.compute(epoch, epoch_time)

    if mode == 'valid':
        torch.set_grad_enabled(True)

    results = print_update(metrics, results, args, mode)
    metric.reset()

    return metrics, results


def print_update(metrics, results, args, mode):
    print('''Epoch: {} Loss: {:.6f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} F1_labeled {} 
             Time: {:.4f}'''.format(metrics['epoch'],
                                    metrics['loss'],
                                    metrics['accuracy'],
                                    metrics['precision'],
                                    metrics['recall'],
                                    metrics['f1_score'],
                                    metrics['f1_score_labels'],
                                    metrics['epoch_time']))

    results = results.append({'model': args.model,
                              'lr': args.lr,
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
