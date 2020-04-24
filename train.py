import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

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

    if args.resume:
        args.mask_use = False
        print('resume model_{}'.format(args.N))
        model, optimizer = create_model(args)
        checkpoint = torch.load(args.model_path + 'model_{}.pt'.format(args.N))
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        input_num = 3 + len(args.attribute)
        if args.model == 'resnet50':
            model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif args.model == 'vgg16':
            model.features[0] = nn.Conv2d(input_num, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        args.mask_use = True
    else:
        model, optimizer = create_model(args)

    if args.show_model:
        print(model)

    model = nn.DataParallel(model)
    model.to(device)

    if args.pos_weight:
        w = {'attribute_globules': args.weights[0],          # 1.2
             'attribute_milia_like_cyst': args.weights[1],   # 1.2
             'attribute_negative_network': args.weights[2],  # 1.5
             'attribute_pigment_network': args.weights[3],   # 0.4
             'attribute_streaks': args.weights[4]}           # 1.5
        pos_weight = torch.Tensor([w[attr] for attr in args.attribute]).to(device)
    else:
        pos_weight = None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # torch.Tensor([0.5, 1.0]).to(device)

    scheduler = MultiStepLR(optimizer, [50,100,140,180], gamma=0.2)

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
            for i, (image_batch, labels_batch, names) in enumerate(trn_dl):
                if i == n_trn - 1:
                    print(f'\r', end='')
                else:
                    print(f'\rBatch {i} / {n_trn} ', end='')

                if not args.aux:
                    image_batch = image_batch.to(device)
                    labels_batch = labels_batch.to(device)
                else:
                    image_batch = image_batch.view(-1, *image_batch.shape[-3:]).to(device)
                    labels_batch = labels_batch.view(-1, labels_batch.shape[-1]).to(device)

                if isinstance(args.attribute, str):
                    labels_batch = torch.reshape(labels_batch, (-1, 1))

                optimizer.zero_grad()
                last_output, aux_output = model(image_batch)
                loss1 = criterion(last_output, labels_batch)
                loss2 = torch.FloatTensor([0.]).to(device)

                if args.aux:
                    for i in range(aux_output.shape[0] // args.aux_batch):
                        l = aux_output[i * args.aux_batch:(i + 1) * args.aux_batch].std(dim=0).data
                        loss2 += torch.mean(l)
                    loss = loss1 + loss2
                else:
                    loss = loss1
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
                    last_output = model(image_batch)
                    if isinstance(args.attribute, str):
                        labels_batch = torch.reshape(labels_batch, (-1, 1))
                    print(last_output.shape)
                    print(labels_batch.shape)
                    loss = criterion(last_output, labels_batch)
                    metrics.valid.update(labels_batch, last_output, loss)
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
