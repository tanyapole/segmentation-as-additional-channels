import json
import h5py
import os


def load_image(file_name):
    f = h5py.File(file_name, 'r')
    file_np = f['img'][()]
    return file_np


def print_save_results(args, results, root, i, time):
    print('Модель {}, предобучение {}, заморозка {}, шаг обучения {}, номер эксперимента {}'.format(args.model,
            args.pretrained, args.freezing, args.lr, args.N))
    root.joinpath('params_{}_num_{}.json'.format(time, i)).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    path = 'Results/{}'.format(time)
    name = 'results.csv'
    file = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    results.to_csv(file, index=False)
