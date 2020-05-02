import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam, SGD

from metrics import Metrics
from models import create_model
from my_dataset import make_loader
from Utils.utils import save_weights, read_split_data, print_update


def jac(y_pred, y_true):
    eps = 1e-15
    # jaccard_target = (targets == 1).float()
    # jaccard_output = torch.sigmoid(outputs)
    intersection = (y_pred * y_true).abs().sum()
    union = y_pred.abs().sum() + y_true.abs().sum()
    return 1 - ((intersection + eps) / (union - intersection + eps))


def train(args, results: pd.DataFrame, SEED: int) -> pd.DataFrame:

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = read_split_data(SEED)

    cudnn.benchmark = True

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    if args.resume:
        #args.mask_use = False
        print('resume model_{}'.format(args.N))
        model = create_model(args)
        checkpoint = torch.load(args.model_path + 'model_{}.pt'.format(args.N))
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        if args.mask_use:
            input_num = 3 + len(args.attribute)
            model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # args.mask_use = True
    else:
        model = create_model(args)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = Adam(parameters, lr=args.lr)
    else:
        optimizer = Adam(parameters, lr=args.lr)

    if args.show_model:
        print(model)

    # model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # writer = SummaryWriter()
    trn_dl, val_dl = make_loader(train_test_id, args, train='train', shuffle=True), \
                     make_loader(train_test_id, args, train='valid', shuffle=False)
    metrics = Metrics(args)
    best_f1 = 0

    for ep in range(1, args.n_epochs):
        try:
            start_time = time.time()
            model.train()
            n_trn = len(trn_dl)
            for i, (image_batch, image_batch_z, labels_batch, names) in enumerate(trn_dl):
                if i == n_trn - 1:
                    print(f'\r', end='')
                else:
                    print(f'\rBatch {i} / {n_trn} ', end='')

                image_batch = image_batch.to(device)
                image_batch_z = image_batch_z.to(device)
                labels_batch = labels_batch.to(device)

                if isinstance(args.attribute, str):
                    labels_batch = torch.reshape(labels_batch, (-1, 1))

                optimizer.zero_grad()
                last_output, aux_output = model(image_batch)

                with torch.no_grad():
                    last_output_z, aux_output_z = model(image_batch_z)
                loss1 = criterion(last_output, labels_batch)

                # loss2 = mse(nn.Sigmoid()(last_output), nn.Sigmoid()(last_output_z))
                loss2 = 0.5 * jac(last_output, last_output_z)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()
                metrics.train.update(labels_batch, last_output, loss, loss1, loss2)
            epoch_time = time.time() - start_time
            computed_metr = metrics.train.compute(ep, epoch_time)
            temp_f1 = computed_metr['f1_score']
            results = print_update(computed_metr, results, args, 'train')
            metrics.train.reset()

            start_time = time.time()
            model.eval()
            with torch.no_grad():
                for i, (image_batch, labels_batch, names) in enumerate(val_dl):
                    image_batch = image_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    last_output, _ = model(image_batch)
                    if isinstance(args.attribute, str):
                        labels_batch = torch.reshape(labels_batch, (-1, 1))
                    loss = criterion(last_output, labels_batch)
                    metrics.valid.update(labels_batch, last_output, loss, loss, torch.cuda.FloatTensor([0.]))
            epoch_time = time.time() - start_time
            computed_metr = metrics.valid.compute(ep, epoch_time)
            results = print_update(computed_metr, results, args, 'valid')
            metrics.valid.reset()

            if args.save_model and ep < 100:
                if temp_f1 > best_f1:
                    name = '{}model_{}.pt'.format(args.model_path, args.N)
                    save_weights(model, name, ep, optimizer)
                    best_f1 = temp_f1

        except KeyboardInterrupt:
            return
    # writer.close()
    return results
