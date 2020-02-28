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

    K_models = 1

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

    bootstrap_models = {}
    optimizers = {}

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    # define models pool
    for i in range(K_models):
        bootstrap_models[i], optimizers[i] = create_model(args, device)

    if args.show_model:
        print(bootstrap_models[0])

    criterion = LossBinary(args.jaccard_weight)

    log = root.joinpath('train.log').open('at', encoding='utf8')
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    scheduler = ReduceLROnPlateau(optimizers[0], 'min', factor=0.8, patience=10, verbose=True)

    writer = SummaryWriter()
    metric = Metrics()

    for ep in range(epoch, args.n_epochs):
        try:
            start_time = time.time()
            for model_id in range(K_models):
                if args.pretrained:
                    if ep == 50:
                        for param in bootstrap_models[model_id].parameters():
                            param.requires_grad = True
                ##################################### training #############################################

                train_loader = make_loader(train_test_id, mask_ind, args, train='train', shuffle=True)
                n1 = len(train_loader)

                for i, (train_image_batch, train_labels_batch, names) in enumerate(train_loader):
                    if i % 50 == 0:
                        print(f'\rBatch {i} / {n1} ', end='')
                    elif i >= n1 - 50:
                        print(f'\r', end='')

                    train_image_batch = train_image_batch.permute(0, 3, 1, 2)
                    train_image_batch = train_image_batch.to(device).type(torch.cuda.FloatTensor)
                    train_labels_batch = train_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[model_id](train_image_batch)

                    if isinstance(args.attribute, str):
                        train_labels_batch = torch.reshape(train_labels_batch, (-1, 1))

                    loss = criterion(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizers[model_id].step()

                    if model_id == 0:
                        outputs = torch.sigmoid(output_probs)
                        metric.update(outputs, train_labels_batch)

            epoch_time = time.time() - start_time

            train_metrics = metric.compute_train(loss, ep, epoch_time)
            print('Epoch: {} Loss: {:.6f} Prec: {:.4f} Recall: {:.4f} Time: {:.4f}'.format(
                                                         ep,
                                                         train_metrics['loss'],
                                                         train_metrics['precision'],
                                                         train_metrics['recall'],
                                                         train_metrics['epoch_time']))

            results = results.append({'freeze_mode': args.freezing,
                                      'lr': args.lr,
                                      'exp': args.N,
                                      'train_mode': 'train',
                                      'epoch': ep,
                                      'loss': train_metrics['loss'],
                                      'prec': train_metrics['precision'],
                                      'recall': train_metrics['recall']}, ignore_index=True)

            metric.reset()
            ##################################### validation ###########################################
            valid_loader = make_loader(train_test_id, mask_ind, args, train='valid', shuffle=True)
            with torch.no_grad():
                n2 = len(valid_loader)
                for i, (valid_image_batch, valid_labels_batch, names) in enumerate(valid_loader):
                    if i == n2-1:
                        print(f'\r', end='')
                    elif i < n2-3:
                        print(f'\rBatch {i} / {n2} ', end='')
                    valid_image_batch = valid_image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)
                    valid_labels_batch = valid_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[0](valid_image_batch)

                    if isinstance(args.attribute, str) and (args.attribute != 'attribute_all'):
                        valid_labels_batch = torch.reshape(valid_labels_batch, (-1, 1))
                    loss = criterion(output_probs, valid_labels_batch)

                    outputs = torch.sigmoid(output_probs)
                    metric.update(outputs, valid_labels_batch)

            valid_metrics = metric.compute_valid(loss)
            print('\t\t Loss: {:.6f} Prec: {:.4f} Recall: {:.4f}'.format(
                                                                 valid_metrics['loss'],
                                                                 valid_metrics['precision'],
                                                                 valid_metrics['recall']))

            results = results.append({'freeze_mode': args.freezing,
                                      'lr': args.lr,
                                      'exp': args.N,
                                      'train_mode': 'valid',
                                      'epoch': ep,
                                      'loss': valid_metrics['loss'],
                                      'prec': valid_metrics['precision'],
                                      'recall': valid_metrics['recall']}, ignore_index=True)
            metric.reset()
            write_event(log, train_metrics=train_metrics, valid_metrics=valid_metrics)
            write_tensorboard(writer, train_metrics, valid_metrics, args)
            scheduler.step(valid_metrics['loss'])


        except KeyboardInterrupt:
            return
    writer.close()
    return results
