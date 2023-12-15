# copyright (c) 2023 Ling-Hao CHEN (https://lhchen.top/) from Tsinghua University.
#
# ERASE is released under License for Non-commercial Scientific Research Purposes.
#
# The ERASE authors team grants you a non-exclusive, worldwide, non-transferable, non-sublicensable, revocable,
# royalty-free, and limited license under the ERASE authors team’s copyright interests to reproduce, distribute,
# and create derivative works of the text, videos, and codes solely for your non-commercial research purposes.
#
# Any other use, in particular any use for commercial, pornographic, military, or surveillance, purposes is prohibited.
#
# Text and visualization results are owned by Ling-Hao CHEN (https://lhchen.top/) from Tsinghua University.
#
#
# ----------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2022 Xiaotian Han
# ----------------------------------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the fllowing open-source project:
# https://github.com/ryanchankh/mcr2/blob/master
# https://github.com/ahxt/G2R
# ----------------------------------------------------------------------------------------------------------------------------
import torch
from utils import label_to_membership


class MaximalCodingRateReduction(torch.nn.Module):
    # MCR2_implement is based on https://github.com/ryanchankh/mcr2/blob/master/loss.py
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    # compute theoretical discrimn loss of learned representations
    def compute_discrimn_loss_theoretical(self, W):
        p, m = W.shape
        I = torch.eye(p)
        I = I.to(W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    # compute empirical discrimn loss of learned representations
    def compute_discrimn_loss_empirical(self, W, device):
        p, m = W.shape
        I = torch.eye(p).to(device)
        I = I.to(W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    # compute theoretical compress loss of learned representations
    def compute_compress_loss_theoretical(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p)
        I = I.to(W.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    # compute empirical compress loss of learned representations
    def compute_compress_loss_empirical(self, W, Pi, device):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2

    def forward(self, X, Y, num_classes):
        device = X.device
        W = X.T

        # generate the membership matrix
        Pi = label_to_membership(Y.cpu().numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(device)

        # compute discrimn loss and compress loss, empirical loss is used for training and theoretical loss is used for justification
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W, device)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(
            W, Pi, device)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        # compute the total loss
        total_loss_empi = -discrimn_loss_empi * \
            self.gam2 + self.gam1 * compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])
