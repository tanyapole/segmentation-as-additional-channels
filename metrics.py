from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np


class Metric:

    def __init__(self, args):

        self.conf_matrix = np.zeros([len(args.attribute), 2, 2])
        self.conf_matrix_s = np.zeros([len(args.attribute), 2, 2])
        self.loss = []
        self.bce1_loss = []
        self.bce2_loss = []

    def update(self, yl_true, yl_pred, ys_true, ys_pred, loss, bce1_loss, bce2_loss):

        yl_pred = (yl_pred.data.cpu().numpy() > 0) * 1
        yl_true = yl_true.data.cpu().numpy()
        loss = loss.detach().cpu().numpy().item()
        bce1_loss = bce1_loss.detach().cpu().numpy().item()
        bce2_loss = bce2_loss.detach().cpu().numpy().item()
        if yl_true.shape[1] == 1:
            self.conf_matrix += confusion_matrix(yl_true, yl_pred, labels=[0, 1])
        else:
            self.conf_matrix += multilabel_confusion_matrix(yl_true, yl_pred)

        ys_true = (ys_true.view(ys_true.shape[0], -1).data.cpu().numpy() > 0) * 1
        ys_pred = ys_pred.view(ys_pred.shape[0], -1).data.cpu().numpy()
        print(ys_pred.shape, ys_true)
        self.conf_matrix_s += multilabel_confusion_matrix(ys_true, ys_pred)
        #print(self.conf_matrix)
        self.loss.append(loss)
        self.bce1_loss.append(bce1_loss)
        self.bce2_loss.append(bce2_loss)

    def compute(self, ep: int, epoch_time: float) -> dict:

        acc_l = []
        prec_l =[]
        rec_l = []
        f1_l  = []
        f1_s  = []
        for cm in self.conf_matrix:
            tn, fp, fn, tp = cm.ravel()
            acc_l.append((tp + tn) / (tp + tn + fp + fn))          # TP+TN/(TP+TN+FP+FN)
            p = tp / (tp + fp + 1e-15)                             # TP   /(TP+FP)
            prec_l.append(p)
            r = tp / (tp + fn + 1e-15)                             # TP   /(TP+FN)
            rec_l.append(r)
            f1_l.append(2 * p * r / (p + r + 1e-15))               # 2*PREC*REC/(PREC+REC)
        acc  = sum(acc_l) / len(acc_l)
        prec = sum(prec_l)/ len(prec_l)
        rec  = sum(rec_l) / len(rec_l)
        f1   = sum(f1_l)  / len(f1_l)

        for cm in self.conf_matrix_s:
            _, fp, fn, tp = cm.ravel()
            p = tp / (tp + fp + 1e-15)                             # TP   /(TP+FP)
            r = tp / (tp + fn + 1e-15)                             # TP   /(TP+FN)
            f1_s.append(2 * p * r / (p + r + 1e-15))               # 2*PREC*REC/(PREC+REC)

        f1_s_mean = sum(f1_s) / len(f1_s)

        loss = sum(self.loss) / len(self.loss)
        bce1_loss = sum(self.bce1_loss) / len(self.bce1_loss)
        bce2_loss = sum(self.bce2_loss) / len(self.bce2_loss)

        return {
                    'epoch': ep,
                    'loss': loss,
                    'bce1_loss': bce1_loss,
                    'bce2_loss': bce2_loss,
                    'epoch_time': epoch_time,
                    'accuracy': acc,
                    'accuracy_labels': acc_l,
                    'precision': prec,
                    'precision_labels': prec_l,
                    'recall' : rec,
                    'recall_labels': rec_l,
                    'f1_score': f1,
                    'f1_score_labels': f1_l,
                    'f1_score_segm': f1_s_mean
                }

    def reset(self):
        self.conf_matrix = np.zeros([self.conf_matrix.shape[0], 2, 2])
        self.conf_matrix_s = np.zeros([self.conf_matrix.shape[0], 2, 2])
        self.loss = []
        self.bce_loss = []
        self.pair_loss = []


class Metrics:

    def __init__ (self, args):
        self.train = Metric(args)
        self.valid = Metric(args)
