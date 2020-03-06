import time
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torch.nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from pathlib import Path
from loss import LossBinary
from Utils.utils import write_event, write_tensorboard
from my_dataset import make_loader
from models import create_model
from metrics import Metrics


def train(args, results):

    epoch = 0

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = pd.read_csv('Data/train_test_id.csv')
    mask_ind = pd.read_csv('Data/mask_ind.csv')

    # uncomment for debugging
    """train_loader = make_loader(train_test_id, mask_ind, args, annotated, train='train', shuffle=True)
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
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    model, optimizer = create_model(args, device)

    if args.show_model:
        print(model)

    criterion = LossBinary(args.jaccard_weight)

    log = root.joinpath('train.log').open('at', encoding='utf8')

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10, verbose=True)

    writer = SummaryWriter()
    metric = Metrics()

    for ep in range(epoch, args.n_epochs):
        try:
            metrics = [0, 0]
            for i, mode in enumerate(['train', 'valid']):

                metrics[i], results = make_step(model=model, mode=mode, train_test_id=train_test_id, mask_ind=mask_ind,
                                                args=args, device=device, criterion=criterion, optimizer=optimizer,
                                                results=results, metric=metric, epoch=ep, scheduler=scheduler)

            write_event(log, metrics=metrics)
            write_tensorboard(writer, metrics=metrics, args=args)

        except KeyboardInterrupt:
            return
    writer.close()
    log.close()
    return results


def make_step(model, mode, train_test_id, mask_ind, args, device, criterion, optimizer, results, metric, epoch,
              scheduler):
    start_time = time.time()
    loader = make_loader(train_test_id, mask_ind, args, train=mode, shuffle=True)
    n = len(loader)
    if mode == 'valid':
        torch.set_grad_enabled(False)
    for i, (image_batch, labels_batch, names) in enumerate(loader):
        if i == n - 1:
            print(f'\r', end='')
        elif i < n - 3:
            print(f'\rBatch {i} / {n} ', end='')
        image_batch = image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)
        labels_batch = labels_batch.to(device).type(torch.cuda.FloatTensor)

        output_probs = model(image_batch)

        if isinstance(args.attribute, str):
            labels_batch = torch.reshape(labels_batch, (-1, 1))
        loss = criterion(output_probs, labels_batch)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        outputs = torch.sigmoid(output_probs)

        outputs = np.around(outputs.data.cpu().numpy().ravel())
        labels_batch = labels_batch.data.cpu().numpy().ravel()
        metric.update(labels_batch, outputs)

    if mode == 'valid':
        torch.set_grad_enabled(True)
        scheduler.step(loss)

    epoch_time = time.time() - start_time

    metrics = metric.compute(loss, epoch, epoch_time)

    print('Epoch: {} Loss: {:.6f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Time: {:.4f}'.format(
        metrics['epoch'],
        metrics['loss'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['epoch_time']))

    results = results.append({'freeze_mode': args.freezing,
                              'lr': args.lr,
                              'exp': args.N,
                              'train_mode': mode,
                              'epoch': epoch,
                              'loss': metrics['loss'],
                              'prec': metrics['precision'],
                              'recall': metrics['recall']}, ignore_index=True)

    metric.reset()

    return metrics, results