from sklearn.metrics import average_precision_score
import numpy as np
from Utils.constants import PRETRAIN


def jaccard(y_true, y_pred):
    intersection = np.sum(np.abs(y_true * y_pred))
    sum_ = np.sum(np.abs(y_true) + np.abs(y_pred))
    jac = (intersection ) / (sum_ - intersection + 1e-10)
    return jac


class Metric:

    def __init__(self, args):

        self.y_true_all = np.array([])
        self.y_pred_all = np.array([])
        self.loss = []

    def update(self, yl_true, yl_pred, loss, train_type: str):

        if train_type != PRETRAIN:
            self.y_pred_all = np.append(self.y_pred_all, yl_pred.data.cpu().numpy())
            self.y_true_all = np.append(self.y_true_all, yl_true.data.cpu().numpy())
        loss = loss.detach().cpu().numpy().item()

        self.loss.append(loss)

    def compute(self, ep: int, epoch_time: float, train_type: str) -> dict:

        if train_type != PRETRAIN:
            metric = average_precision_score(y_true=self.y_true_all, y_score=self.y_pred_all, average='micro')
        else:
            metric = 0.
        loss = sum(self.loss) / len(self.loss)
        return {
                    'epoch': ep,
                    'loss': loss,
                    'epoch_time': epoch_time,
                    'metric': metric
                }

    def reset(self):
        self.y_true_all = np.array([])
        self.y_pred_all = np.array([])
        self.loss = []


class Metrics:

    def __init__ (self, args):
        self.train = Metric(args)
        self.valid = Metric(args)
