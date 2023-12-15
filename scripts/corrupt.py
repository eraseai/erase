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
# MIT License
# Copyright (c) 2022 Xiaotian Han
# ----------------------------------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the fllowing open-source project:
# https://github.com/ryanchankh/mcr2/blob/master
# https://github.com/ahxt/G2R
# ----------------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch

def default_corrupt(trainset, ratio, seed, train_mask, t="asymm"):
    """
    Corrupt labels in trainset.
    
    Args:
        trainset (torch.data.dataset): trainset with clean labels
        split_idx (torch.tensor): index of training set
        ratio (float): corruption ratio
        seed (int): random seed
        t (str): type of corruption
        
    Returns:
        label (torch.tensor): corrupted labels
        
    """
    if t == "asymm":
        #initialization
        label = []
        num_classes = np.max(trainset.y.cpu().numpy()) + 1
        np.random.seed(seed)

        #generate the corruption matrix
        C = np.eye(num_classes) * (1 - ratio)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            C[i][np.random.choice(row_indices[row_indices != i])] = ratio

        #corrupt the labels and append them into a list
        for label_i in trainset.y[train_mask]:
            data1 = np.random.choice(trainset.num_classes, p=C[label_i])
            label.append(data1)
        label = torch.tensor(label).long()

        return label
    elif t=="symm":
        #initialization
        label = []
        num_classes = np.max(trainset.y.cpu().numpy()) + 1
        np.random.seed(seed)

        #generate the corruption matrix
        off_diagnal= ratio * np.full((num_classes, num_classes), 1 / (num_classes-1))
        np.fill_diagonal(off_diagnal, 0)
        data = np.eye(num_classes) * (1 - ratio) + off_diagnal

        #corrupt the labels and append them into a list
        for label_i in trainset.y[train_mask]:
            data1 = np.random.choice(num_classes, p=data[label_i])
            label.append(data1)
        label = np.array(label)
        label = torch.tensor(label).long()
        
        return label