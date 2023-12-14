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
import torch
from cuml.preprocessing import normalize
from cuml.linear_model import LogisticRegression


def Linear_classifier(args, out,split_idx,noisy_train_labels,clean_labels,evaluator):
    """
    Evaluate the representation quality of the model using logistic regression.

    Args:
        out: Learned representations.
        split_idx: The split index.
        clean_labels: Ground-truth node labels.
        evaluator: The evaluator from ogb.nodeproppred.

    Returns:
        train_acc: Accuracy on training set.
        valid_acc: Accuracy on validation set.
        test_acc: Accuracy on test set.
    """
    #detach the output from the graph and normalize it
    out = out.detach().cpu().numpy()
    out = normalize(out,norm='l2')

    #train a logistic regression classifier using the representations
    train_features = out[split_idx['train']]
    clf = LogisticRegression(solver='qn', max_iter=500).fit(train_features, noisy_train_labels.ravel())
    y_pred = clf.predict_proba(out).argmax(axis=-1, keepdims=True) 

    #use the evaluator from ogb.nodeproppred to compute the accuracy of the classifier in order to measure the representation quality
    train_acc = evaluator.eval({
        'y_true': clean_labels[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': clean_labels[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': clean_labels[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc,valid_acc,test_acc

    
