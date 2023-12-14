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
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import label_to_membership,compute_sematic_labels



# MCR2_implement is based on https://github.com/ryanchankh/mcr2/blob/master/loss.py
class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01,corafull=False):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.corafull = corafull
    
    #compute theoretical discrimn loss of learned representations
    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p)
        I = I.to(W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    #compute empirical discrimn loss of learned representations
    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.
    
    #compute theoretical compress loss of learned representations
    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss
    
    #compute empirical compress loss of learned representations
    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    #optimized function to compute empirical compress loss of learned representations
    #using mask to replace membership matrix to reduce memory consumption for cases where the dataset has many classes
    def compute_compress_loss_empirical_manyclass(self, W, L_all):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k = torch.max(L_all)+1
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            mask = torch.where(L_all == j)[0]
            W_masked = W[:,mask]
            trPi = W_masked.shape[1] + 1e-8
            scalar = p / (trPi * self.eps)
            WWT = W_masked.matmul(W_masked.T)
            padding = (0, p - WWT.shape[0], 0, p - WWT.shape[0])
            padded = F.pad(WWT, padding)
            log_det = torch.logdet(I + scalar * padded)
            compress_loss += log_det * trPi / m
        return compress_loss / 2.
    

    def forward_ERASEmanyclass(self,X,data,A,Y_all,alpha,beta,T,train_mask):
        """
        Compute the maximal coding rate reduction loss for small dataset that have lots of classes.
        Calculations using membership matrix is replaced by masking W, which reduces memory consumption.

        Args:
            X: The input data.
            A: The adjacency matrix.
            Y_all: The semantic labels.
            alpha: The parameter that controls the influence of structural information.
            beta: The parameter that controls the influence of semantic information.
            T: The number of propagation steps when calculating structural information.
            train_mask: The train mask.

        Returns:
            total_loss_empi: The empirical loss.
            [discrimn_loss_empi, compress_loss_empi]: The empirical loss of discriminative and compressive terms. Theoretical loss is not calculated to reduce memory consumption.
            L_all: The predicted labels.
        """
        #initialization
        num_classes = data.y.max().item() + 1
        X_train = X[train_mask]
        W = X.T

        #compute the robust labels for training using semantic information
        Y_all = compute_sematic_labels(X_train,Y_all,X,train_mask,beta,num_classes)

        #propagate the labels to add structural information
        for i in range(T):
            Y_all = (1-alpha)*torch.matmul(A,Y_all) + alpha*Y_all
        
        #compute the predicted labels, generate the membership matrix
        L_all = torch.argmax(Y_all,dim=1)

        #compute discrimn loss and compress loss, theoretical loss is skipped to reduce memory consumption
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical_manyclass(W, L_all)

        #compute the total loss
        total_loss_empi = - self.gam2 * discrimn_loss_empi + self.gam1 * compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],L_all)

    def forward_ERASE(self, X,data,A,Y_all,alpha,beta,T,train_mask):
        """
        Compute the maximal coding rate reduction loss.

        Args:
            X: The input data.
            A: The adjacency matrix.
            Y_all: The semantic labels.
            alpha: The parameter that controls the influence of structural information.
            beta: The parameter that controls the influence of semantic information.
            T: The number of propagation steps when calculating structural information.
            train_mask: The train mask.

        Returns:
            total_loss_empi: The empirical loss.
            [discrimn_loss_empi, compress_loss_empi]: The empirical loss of discriminative and compressive terms.
            [discrimn_loss_theo, compress_loss_theo]: The theoretical loss of discriminative and compressive terms.
            L_all: The predicted labels.
        """
        #initialization
        num_classes = data.y.max().item() + 1
        X_train = X[train_mask]
        W = X.T

        #compute the robust labels for training using semantic information
        Y_all = compute_sematic_labels(X_train,Y_all,X,train_mask,beta,num_classes)

        #propagate the labels to add structural information
        for i in range(T):
            Y_all = (1-alpha)*torch.matmul(A,Y_all) + alpha*Y_all
        
        #compute the predicted labels, generate the membership matrix
        L_all = torch.argmax(Y_all,dim=1)
        Pi = label_to_membership(L_all.cpu(),num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(X.device)

        #compute discrimn loss and compress loss, empirical loss is used for training and theoretical loss is used for justification
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        #compute the total loss
        total_loss_empi = - self.gam2 * discrimn_loss_empi + self.gam1 * compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()],L_all)
    
    def forward(self, X,data,A,Y,alpha,beta,T,train_mask):
        if not self.corafull:
            return self.forward_ERASE(X,data,A,Y,alpha,beta,T,train_mask)
        elif self.corafull:
            return self.forward_ERASEmanyclass(X,data,A,Y,alpha,beta,T,train_mask)