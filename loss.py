from torch import nn


class LossBinary:

    def __init__(self, pos_weight=None):
        self.nll_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, last_output, aux_output, targets):

        return self.nll_loss(last_output, targets)
