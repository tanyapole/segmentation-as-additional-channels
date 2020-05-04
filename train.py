import time

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


def train(args, results: pd.DataFrame, SEED: int) -> pd.DataFrame:

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = read_split_data(SEED)

    cudnn.benchmark = True

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    model = create_model(args)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
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

    for ep in range(0, args.n_epochs):
        try:
            start_time = time.time()
            model.train()
            n_trn = len(trn_dl)
            for i, (image_batch, masks_batch, labels_batch, names) in enumerate(trn_dl):
                if i == n_trn - 1:
                    print(f'\r', end='')
                else:
                    print(f'\rBatch {i} / {n_trn} ', end='')

                image_batch = image_batch.to(device)
                masks_batch = masks_batch.to(device)
                labels_batch = labels_batch.to(device)

                if isinstance(args.attribute, str):
                    labels_batch = torch.reshape(labels_batch, (-1, 1))

                optimizer.zero_grad()
                clsf_output, segm_output = model(image_batch)
                # print(clsf_output.shape, segm_output.shape, masks_batch.shape)
                loss1 = criterion(clsf_output, labels_batch)
                loss2 = criterion(segm_output, masks_batch)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()
                metrics.train.update(labels_batch, clsf_output, loss, loss1, loss2)
            epoch_time = time.time() - start_time
            computed_metr = metrics.train.compute(ep, epoch_time)
            temp_f1 = computed_metr['f1_score']
            results = print_update(computed_metr, results, args, 'train')
            metrics.train.reset()

            start_time = time.time()
            model.eval()
            with torch.no_grad():
                for i, (image_batch, masks_batch, labels_batch, names) in enumerate(val_dl):
                    image_batch = image_batch.to(device)
                    masks_batch = masks_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    clsf_output, segm_output = model(image_batch)
                    if isinstance(args.attribute, str):
                        labels_batch = torch.reshape(labels_batch, (-1, 1))
                    loss1 = criterion(clsf_output, labels_batch)
                    loss2 = criterion(segm_output, masks_batch)
                    loss = loss1 + loss2
                    metrics.valid.update(labels_batch, clsf_output, loss, loss1, loss2)
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
