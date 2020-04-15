import time
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.backends import cudnn
from pathlib import Path
from my_dataset import make_loader
from models import create_model
from metrics import Metrics
import random


def train(args, results, best_f1, seed):

    epoch = 0

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = pd.read_csv('Data/train_test_id_with_masks.csv')
    indexes = np.arange(train_test_id.shape[0])
    random.seed(seed)
    random.shuffle(indexes)
    train_test_id = train_test_id.iloc[indexes].reset_index(drop=True)
    #train_test_id = train_test_id.sample(frac=1).reset_index(drop=True)
    train_test_id.loc[:1900, 'Split'] = 'train'
    train_test_id.loc[1900:, 'Split'] = 'valid'

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

    writer = SummaryWriter()
    metric = Metrics(args)

    for ep in range(epoch, args.n_epochs):
        try:
            metrics = [0, 0]
            for i, mode in enumerate(['train', 'valid']):
                metrics[i], results = make_step(model=model, mode=mode, train_test_id=train_test_id,
                                                args=args, device=device, criterion=criterion,
                                                optimizer=optimizer, results=results, metric=metric, epoch=ep)
            # write_tensorboard(writer, metrics=metrics, args=args)

        except KeyboardInterrupt:
            return
    writer.close()
    return results, best_f1


def make_step(model, mode, train_test_id, args, device, criterion, optimizer, results, metric, epoch,
              scheduler):
    start_time = time.time()
    loader = make_loader(train_test_id, args, train=mode, shuffle=True)
    n = len(loader)
    if mode == 'valid':
        torch.set_grad_enabled(False)
    for i, (image_batch, labels_batch, names) in enumerate(loader):
        if i == n - 1:
            print(f'\r', end='')
        elif i < n - 3:
            print(f'\rBatch {i} / {n} ', end='')
        """if i < 5 :
            fig = plt.figure(figsize=(10, 10))
            ax = []
            for i, image in enumerate(image_batch):
                for channel in range(3, image.shape[2]):
                    im = image.cpu().numpy()[:, :, channel]
                    ax.append(fig.add_subplot(len(image_batch), image.shape[2], i*(image.shape[2]-3)+channel-3 + 1))
                    ax[i*(image.shape[2]-3)+channel-3].set_title(str(np.unique(im))) #names[i][5:]+
                    plt.imshow(im)
            plt.show()"""

        image_batch = image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)
        last_output = model(image_batch)
        labels_batch = labels_batch.to(device).type(torch.cuda.FloatTensor)

        if isinstance(args.attribute, str):
            labels_batch = torch.reshape(labels_batch, (-1, 1))

        outputs = nn.Sigmoid()(last_output)
        #loss = criterion(last_output, labels_batch)
        loss = nn.BCELoss()(outputs, labels_batch)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print(labels_batch)
        #print(outputs)
        outputs = np.around(outputs.data.cpu().numpy())
        labels_batch = labels_batch.data.cpu().numpy()
        #print(outputs, labels_batch)
        metric.update(labels_batch, outputs, loss, loss, torch.Tensor([0]))

    epoch_time = time.time() - start_time

    metrics = metric.compute(epoch, epoch_time)

    if mode == 'valid':
        torch.set_grad_enabled(True)
        scheduler.step(loss)

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

    results = results.append({'mask_use': args.mask_use,
                              'aux': args.aux,
                              'aux_batch': args.aux_batch,
                              'freeze_mode': args.freezing,
                              'lr': args.lr,
                              'exp': args.N,
                              'cell': args.cell,
                              'cell_size': args.cell_size,
                              'prob': args.prob,
                              'train_mode': mode,
                              'epoch': metrics['epoch'],
                              'loss': metrics['loss'],
                              'bce_loss': metrics['bce_loss'],
                              'std_loss': metrics['std_loss'],
                              'acc': metrics['accuracy'],
                              'acc_labels': metrics['accuracy_labels'],
                              'prec': metrics['precision'],
                              'prec_labels': metrics['precision_labels'],
                              'recall': metrics['recall'],
                              'recall_labels': metrics['recall_labels'],
                              'f1': metrics['f1_score'],
                              'f1_labels': metrics['f1_score_labels']}, ignore_index=True)

    return results
