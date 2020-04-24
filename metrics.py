from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np


class Metric:

    def __init__(self, args):

        self.conf_matrix = np.zeros([len(args.attribute), 2, 2])
        self.loss = []
        self.bce_loss = []
        self.std_loss = []

    def update(self, y_true, y_pred, loss, bce_loss, std_loss):

        y_pred = (y_pred.data.cpu().numpy() > 0) * 1
        y_true = y_true.data.cpu().numpy()
        loss = loss.detach().cpu().numpy().item()
        if y_true.shape[1] == 1:
            self.conf_matrix += confusion_matrix(y_true, y_pred, labels=[0, 1])
        else:
            self.conf_matrix += multilabel_confusion_matrix(y_true, y_pred)
        #print(self.conf_matrix)
        self.loss.append(loss)
        self.bce_loss.append(bce_loss)
        self.std_loss.append(std_loss)

    def compute(self, ep, epoch_time):

        acc_l = []
        prec_l =[]
        rec_l = []
        f1_l  = []
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
        loss = sum(self.loss) / len(self.loss)
        bce_loss = sum(self.bce_loss) / len(self.bce_loss)
        std_loss = sum(self.std_loss) / len(self.std_loss)

        return {
                    'epoch': ep,
                    'loss': loss,
                    'bce_loss': bce_loss,
                    'std_loss': std_loss,
                    'epoch_time': epoch_time,
                    'accuracy': acc,
                    'accuracy_labels': acc_l,
                    'precision': prec,
                    'precision_labels': prec_l,
                    'recall' : rec,
                    'recall_labels': rec_l,
                    'f1_score': f1,
                    'f1_score_labels': f1_l
                }

    def reset(self):
        self.conf_matrix = np.zeros([self.conf_matrix.shape[0], 2, 2])
        self.loss = []
        self.bce_loss = []
        self.std_loss = []


class Metrics:

    def __init__ (self, args):
        self.train = Metric(args)
        self.valid = Metric(args)
