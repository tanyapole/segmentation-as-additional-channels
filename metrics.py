from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, average_precision_score
import numpy as np
from Utils.constants import ALL_ATTRIBUTES

class Metric:

    def __init__(self, args):

        self.y_true_all = np.array([])
        self.y_pred_all = np.array([])
        self.loss = []

    def update(self, yl_true, yl_pred, loss):

        self.y_pred_all = np.append(self.y_pred_all, yl_pred.data.cpu().numpy())
        self.y_true_all = np.append(self.y_true_all, yl_true.data.cpu().numpy())
        loss = loss.detach().cpu().numpy().item()

        self.loss.append(loss)

    def compute(self, ep: int, epoch_time: float) -> dict:

        AP = average_precision_score(y_true=self.y_true_all, y_score=self.y_pred_all, average='micro')
        loss = sum(self.loss) / len(self.loss)
        return {
                    'epoch': ep,
                    'loss': loss,
                    'epoch_time': epoch_time,
                    'AP': AP
                }

    def reset(self):
        self.y_true_all = np.array([])
        self.y_pred_all = np.array([])
        self.loss = []


class Metrics:

    def __init__ (self, args):
        self.train = Metric(args)
        self.valid = Metric(args)
