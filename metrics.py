from sklearn.metrics import confusion_matrix


class Metrics:

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, y_true, y_pred):

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self, loss, ep, epoch_time):
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)  # TP + TN / (TP + TN + FP + FN)
        prec = self.tp / (self.tp + self.fp)  # TP / (TP + FP)
        rec = self.tp / (self.tp + self.fn)  # TP / (TP + FN)
        f1 = 2 * prec * rec / (rec + prec)  # 2 * PREC * REC / (PREC + REC)
        return {
                    'epoch': ep,
                    'loss': loss.item(),
                    'epoch_time': epoch_time,
                    'accuracy': acc,
                    'precision': prec,
                    'recall' : rec,
                    'f1_score': f1
                }

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
