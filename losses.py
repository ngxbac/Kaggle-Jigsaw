import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, loss_weight):
        super(CustomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.bin_loss = nn.BCEWithLogitsLoss
        self.aux_loss = nn.BCEWithLogitsLoss

    def forward(self, output_bin, target_bin, weight_bin, output_aux, target_aux):
        bin_loss = self.bin_loss(weight=weight_bin)(output_bin, target_bin)
        aux_loss = self.aux_loss()(output_aux, target_aux)
        return self.loss_weight * bin_loss + aux_loss
