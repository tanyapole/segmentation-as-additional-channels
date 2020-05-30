from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np


class Metric:

    def __init__(self, args):

        self.conf_matrix = np.zeros([len(args.attribute), 2, 2])
        self.loss = []

    def update(self, yl_true, yl_pred, loss):

        yl_pred = (yl_pred.data.cpu().numpy() > 0) * 1
        yl_true = yl_true.data.cpu().numpy()
        loss = loss.detach().cpu().numpy().item()

        if yl_true.shape[1] == 1:
            self.conf_matrix += confusion_matrix(yl_true, yl_pred, labels=[0, 1])
        else:
            self.conf_matrix += multilabel_confusion_matrix(yl_true, yl_pred)

        self.loss.append(loss)

    def compute(self, ep: int, epoch_time: float) -> dict:

        acc_l = []

        for cm in self.conf_matrix:
            tn, fp, fn, tp = cm.ravel()
            acc_l.append((tp + tn) / (tp + tn + fp + fn))          # TP+TN/(TP+TN+FP+FN)

        acc  = sum(acc_l) / len(acc_l)
        loss = sum(self.loss) / len(self.loss)

        return {
                    'epoch': ep,
                    'loss': loss,
                    'epoch_time': epoch_time,
                    'accuracy': acc,
                    'accuracy_labels': acc_l
                }

    def reset(self):
        self.conf_matrix = np.zeros([self.conf_matrix.shape[0], 2, 2])
        self.loss = []


class Metrics:

    def __init__ (self, args):
        self.train = Metric(args)
        self.valid = Metric(args)
