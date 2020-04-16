import json
import h5py
import torch
import os


def load_image(file_name):
    f = h5py.File(file_name, 'r')
    file_np = f['img'][()]
    return file_np


def print_save_results(args, results, root, i, time):
    print('Модель {}, mask usage {}, заморозка {}, lr {}, номер эксперимента {} размер квадрата {}, вероятность {}'
          .format(args.model, args.mask_use, args.freezing, args.lr, args.N, args.cell_size, args.prob))
    root.joinpath('params_{}_num {} .json'.format(time, i)).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    path = 'Results/{}'.format(time)
    name = 'results.csv'
    file = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    results.to_csv(file, index=False)


def save_weights(model, model_path, metrics, optimizer):
    torch.save({'model': model.module.state_dict(),
                'epoch_time': metrics[0]['epoch'],
                'valid_loss': metrics[1]['loss'],
                'train_loss': metrics[0]['loss'],
                'optimizer': optimizer.state_dict()},
               str(model_path)
               )
