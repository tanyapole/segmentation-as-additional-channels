import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.backends import cudnn
from torch.optim import Adam

from metrics import Metrics
from models import create_model
from my_dataset import make_loader
from Utils.constants import YNET, PRETRAIN
from Utils.utils import save_weights, read_split_data, print_update


def train(args, results: pd.DataFrame, SEED: int, train_type: str, epochs: int) -> pd.DataFrame:

    train_test_id = read_split_data(SEED, train_type)

    cudnn.benchmark = True

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    model = create_model(args, train_type)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.show_model:
        print(model)

    if not args.cuda1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    trn_dl, val_dl = make_loader(train_test_id, args, train_type=train_type, train='train', shuffle=True), \
                     make_loader(train_test_id, args, train_type=train_type, train='valid', shuffle=False)
    metrics = Metrics(args)
    best_loss = 10**15
    for ep in range(epochs):
        try:
            start_time = time.time()
            model.train()
            n_trn = len(trn_dl)
            for i, (image_batch, target_batch, names) in enumerate(trn_dl):
                if i == n_trn - 1:
                    print(f'\r', end='')
                else:
                    print(f'\rBatch {i} / {n_trn} ', end='')

                image_batch = image_batch.to(device)
                target_batch = target_batch.to(device)

                optimizer.zero_grad()

                output = model(image_batch)
                loss = criterion(output, target_batch)
                loss.backward()
                optimizer.step()
                metrics.train.update(output, target_batch, loss=loss, train_type=train_type)
            epoch_time = time.time() - start_time
            computed_metr = metrics.train.compute(ep, epoch_time, train_type=train_type)
            results = print_update(computed_metr, results, args, 'train', train_type)
            metrics.train.reset()

            start_time = time.time()
            model.eval()
            with torch.no_grad():
                for i, (image_batch, target_batch, names) in enumerate(val_dl):
                    image_batch = image_batch.to(device)
                    target_batch = target_batch.to(device)

                    output = model(image_batch)
                    loss = criterion(output, target_batch)

                    metrics.valid.update(output, target_batch, loss=loss, train_type=train_type)
            epoch_time = time.time() - start_time
            computed_metr = metrics.valid.compute(ep, epoch_time, train_type=train_type)
            temp_loss = computed_metr['loss']
            results = print_update(computed_metr, results, args, 'valid', train_type)
            metrics.valid.reset()

            if ep == epochs -1:
                name = '{}model_base_{}.pt'.format(args.model_path, args.N)
                save_weights(model, name, ep, optimizer)
            if train_type == PRETRAIN:
                if temp_loss < best_loss:
                    name = '{}model_{}.pt'.format(args.model_path, args.N)
                    save_weights(model, name, ep, optimizer)
                    best_loss = temp_loss
            if train_type == YNET and ep == epochs - 1:
                name = '{}model_ynet_{}.pt'.format(args.model_path, args.N)
                save_weights(model, name, ep, optimizer)
        except KeyboardInterrupt:
            return
    return results
