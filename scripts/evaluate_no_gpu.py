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
from utils import sort_dataset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

def accuracy(clf,features,labels):
    y_pred = clf.predict(features)
    acc = accuracy_score(labels, y_pred)
    return acc

def Linear_classifier(args,data,features, noisy_train_labels,clean_labels):
    """
    Evaluate the representation quality of the model using logistic regression.

    Args: 
        data: the dataset
        features: the learned representations
        noisy_train_labels: semantic labels of the training set
        clean_labels: ground truth labels of the dataset
    
    Returns:
        train_acc: the accuracy of training set
        val_acc: the accuracy of validation set
        test_acc: the accuracy of test set
    """
    #detach the output from the graph and normalize it
    noisy_train_labels = noisy_train_labels.cpu().numpy()
    clean_labels = clean_labels.cpu().numpy()
    features = normalize(features.cpu().numpy(),norm='l2')
    train_features = features[data.train_mask.cpu().numpy()]
    val_features = features[data.val_mask.cpu().numpy()]
    test_features = features[data.test_mask.cpu().numpy()]

    #train a logistic regression classifier using the representations
    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000).fit(train_features, noisy_train_labels.ravel())

    #Compute the accuracy of the classifier in order to measure the representation quality
    train_acc = accuracy(clf,train_features, clean_labels[data.train_mask.cpu().numpy()])
    val_acc = accuracy(clf,val_features, clean_labels[data.val_mask.cpu().numpy()])
    test_acc = accuracy(clf,test_features, clean_labels[data.test_mask.cpu().numpy()])
    
    return train_acc,val_acc,test_acc
    
