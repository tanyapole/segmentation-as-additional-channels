from sklearn.metrics import confusion_matrix


class Metrics:

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.loss = []
        self.bce_loss = []
        self.std_loss = []

    def update(self, y_true, y_pred, loss, bce_loss, std_loss):

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        print(tn, fp, fn, tp)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.loss.append(loss.item())
        self.bce_loss.append(bce_loss.item())
        self.std_loss.append(std_loss.item())

    def compute(self, ep, epoch_time):
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)  # TP + TN / (TP + TN + FP + FN)
        prec = self.tp / (self.tp + self.fp)  # TP / (TP + FP)
        rec = self.tp / (self.tp + self.fn)  # TP / (TP + FN)
        f1 = 2 * prec * rec / (rec + prec)  # 2 * PREC * REC / (PREC + REC)
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
                    'precision': prec,
                    'recall' : rec,
                    'f1_score': f1
                }

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.loss = []
        self.bce_loss = []
        self.std_loss = []
