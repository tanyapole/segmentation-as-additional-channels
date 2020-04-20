import time

import torch
from torch.backends import cudnn
from pathlib import Path

from my_dataset import make_loader
from models import create_model
from metrics import Metrics
from Utils.utils import save_weights, read_split_data, print_update


def train(args, results, SEED):

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = read_split_data(SEED)

    # uncomment for debugging
    train_loader = make_loader(train_test_id, args, train='train', shuffle=True)
    print('--' * 10)
    print('check data')
    train_image, train_labels_ind, name = next(iter(train_loader))
    print('train_image.shape', train_image.shape)
    print('train_label_ind.shape', train_labels_ind.shape)
    print('train_image.min', train_image.min().item())
    print('train_image.max', train_image.max().item())
    print('train_label_ind.min', train_labels_ind.min().item())
    print('train_label_ind.max', train_labels_ind.max().item())
    print('--' * 10)

    cudnn.benchmark = True
    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    model, optimizer = create_model(args)

    if args.show_model:
        print(model)

    # model = nn.DataParallel(model)
    model.to(device)
    model = torch.nn.DataParallel(model)
    criterion = torch.nn.BCEWithLogitsLoss()
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
                print(f'\rBatch {i} / {n_trn} ', end='')

                image_batch = image_batch.to(device)
                labels_batch = labels_batch.to(device)

                if isinstance(args.attribute, str):
                    labels_batch = torch.reshape(labels_batch, (-1, 1))

                optimizer.zero_grad()
                last_output = model(image_batch)
                loss = criterion(last_output, labels_batch)
                loss.backward()
                optimizer.step()

                metrics.train.update(labels_batch, last_output, loss)

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
                    loss = criterion(last_output, labels_batch)
                    metrics.valid.update(labels_batch, last_output, loss)
            epoch_time = time.time() - start_time
            computed_metr = metrics.valid.compute(ep, epoch_time)
            results = print_update(computed_metr, results, args, 'valid')
            metrics.valid.reset()

            if args.save_model and ep < 100:
                if temp_f1 > best_f1:
                    name = '{}model_{}.pt'.format(args.model_path, args.N)
                    save_weights(model, name, optimizer)
                    best_f1 = temp_f1

        except KeyboardInterrupt:
            return
    return results
