from torch import nn


class LossBinaryWithAux:

    def __init__(self, pos_weight):
        self.nll_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.MSE = nn.MSELoss()

    def __call__(self, last_output, aux_output, targets):

        loss = self.nll_loss(last_output, targets)
        loss2 = self.MSE(aux_output)

        return loss + loss2
