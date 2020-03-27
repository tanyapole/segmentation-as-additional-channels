import time
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from loss import LossBinaryWithAux
from Utils.utils import write_tensorboard, save_weights
from my_dataset import make_loader
from models import create_model
from metrics import Metrics


def train(args, results, best_f1):

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

    if args.resume:
        args.mask_use = False
        model, optimizer = create_model(args, device)
        checkpoint = torch.load('model.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        input_num = 3 + len(args.attribute)
        if args.model == 'resnet50':
            model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        args.mask_use = True
    else:
        model, optimizer = create_model(args, device)

    if args.show_model:
        print(model)

    model = nn.DataParallel(model)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5, 1.0]).to(device))

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
            if metrics[0]['f1_score'] > best_f1:
                if args.resume:
                    name = 'resume_model.pt'
                else:
                    name = 'model.pt'
                save_weights(model, name, metrics, optimizer)
                best_f1 = metrics[0]['f1_score']
            # write_tensorboard(writer, metrics=metrics, args=args)

        except KeyboardInterrupt:
            return
    writer.close()
    return results, best_f1


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

        if mode == 'train':
            image_batch = image_batch[0]
            labels_batch = labels_batch[0]

        image_batch = image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)

        labels_batch = labels_batch.to(device).type(torch.cuda.FloatTensor)
        last_output, aux_output = model(image_batch)

        if isinstance(args.attribute, str):
            labels_batch = torch.reshape(labels_batch, (-1, 1))

        loss1 = criterion(last_output, labels_batch)

        if mode == 'train':
            l = aux_output.std(dim=0).data
            loss2 = torch.mean(l)
            loss = loss1 + loss2
        else:
            loss = loss1

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        outputs = torch.sigmoid(last_output)

        outputs = np.around(outputs.data.cpu().numpy().ravel())
        labels_batch = labels_batch.data.cpu().numpy().ravel()
        metric.update(labels_batch, outputs)

    epoch_time = time.time() - start_time

    metrics = metric.compute(loss, epoch, epoch_time)

    if mode == 'valid':
        torch.set_grad_enabled(True)
        scheduler.step(loss)

    print('Epoch: {} Loss: {:.6f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Time: {:.4f}'.format(
        metrics['epoch'],
        metrics['loss'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['epoch_time']))

    results = results.append({'mask_use': args.mask_use,
                              'freeze_mode': args.freezing,
                              'lr': args.lr,
                              'exp': args.N,
                              'cell': args.cell,
                              'cell_size': args.cell_size,
                              'prob': args.prob,
                              'train_mode': mode,
                              'epoch': epoch,
                              'loss': metrics['loss'],
                              'acc': metrics['accuracy'],
                              'prec': metrics['precision'],
                              'f1':metrics['f1_score'],
                              'recall': metrics['recall']}, ignore_index=True)

    metric.reset()

    return metrics, results
