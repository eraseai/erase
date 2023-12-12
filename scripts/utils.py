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
import torch.nn
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T
import logging
import os
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import LabelPropagation
from torch_geometric.utils import to_dense_adj

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

def normalize_adj_row(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    np.seterr(divide='ignore')
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    np.seterr(divide='warn')
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).tocoo()

def label_to_membership(targets, num_classes=None):
    """Convert labels to membership matrix."""
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    max_indices = np.argmax(targets, axis=1)
    Pi[max_indices, np.arange(num_samples), np.arange(num_samples)] = 1
    return Pi

def compute_sematic_labels(X_train,Y_all,X,train_mask,beta,num_classes):
    """
    Compute robust labels using semantic information

    Args:
        X_train: The input data.
        Y_all: The semantic labels.
        X: The input data.
        train_mask: The train mask.
        beta: The parameter that controls the influence of structural information.
        num_classes: The number of classes.

    Returns:
        Y_all: The robust labels.
    """
    #initialization
    device = X.device
    Y_train = Y_all[train_mask,:].argmax(dim=1)

    #sort the data by labels
    sorted_x , sorted_y = sort_dataset(X_train,Y_train,num_classes=num_classes,stack=False)
    mvs = np.vstack([np.mean(sorted_x[i],axis=0) for i in range(num_classes)])

    #compute the cosine similarity to add semantic information
    cos_mat = cosine_similarity(X.detach().cpu().numpy(),mvs)
    cos_mat = torch.tensor(np.abs(cos_mat)).to(device) 
    Y_all = beta*Y_all + (1-beta)*cos_mat

    return F.softmax(Y_all,dim=1)

def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset by labels."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        lbl = int(lbl)
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def index_to_mask(index, size):
    """Convert index to mask."""
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_coauthor_amazon_splits(data, num_classes, lcc_mask = None):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

def get_dataset(args):
    """Get dataset."""
    path = 'data'
    if args.dataset == "Cora":
        path = os.path.join(path,'Planetoid')
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif args.dataset == "CiteSeer":
        path = os.path.join(path,'Planetoid')
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Planetoid(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif args.dataset == "PubMed":
        path = os.path.join(path,'Planetoid')
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Planetoid(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif args.dataset == "CoraFull":
        path = os.path.join(path,'Citationfull')
        dataset = CitationFull(path, "cora")
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    else:
        print("Input dataset name!!")
        raise NotImplementedError
    return data

def setup_logger(logger, log_file, level=logging.DEBUG):
    """Set up logger."""
    logger.setLevel(level)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def setup_seed(seed):
    """Set up seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['SEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def denoising_propagation(args,noisy_y):
    """Propagate the noisy labels."""
    device = args.device
    dataset = PygNodePropPredDataset(name=args.dataset,root='dataset', transform=T.Compose([
        T.ToUndirected(),]))
    data = dataset[0].to(device)
    split_idx = dataset.get_idx_split()
    model = LabelPropagation(num_layers=args.T, alpha=args.alpha)
    out =  model(noisy_y, data.edge_index, mask=split_idx['train'].to(device))
    return out    

def sort_dataset_large(data, labels, num_classes=10, stack=False):
    """Sort large-scale dataset by labels."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    sorted_data = [[] for _ in range(num_classes)]
    index = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        lbl = int(lbl)
        sorted_data[lbl].append(data[i])
        index[lbl].append(i)
    for i in range(num_classes):
        if len(sorted_data[i]) == 0:
            sorted_data[i].append([])
            index[i].append([])
        else:
            sorted_data[i] = np.stack(sorted_data[i])
            index[i] = np.stack(index[i])
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
        index = np.hstack(index) 

    return sorted_data, sorted_labels,index

@torch.no_grad()
def compute_semantic_structral_label(args,data,model,loader,Y,train_idx):
    """
    Compute robust labels using semantic and structural information. Used in train_main_arxiv.py.
    
    Args:
        data: The input data.
        model: The model.
        loader: The subgraph loader.
        Y: The semantic labels.
        train_idx: The train mask.
        args: The arguments.
    
    Returns:
        out: The robust labels.
    """
    #initialization
    model.eval()
    num_classes = args.num_classes
    device = args.device

    #compute a temporary output to sort the data by labels
    Z_all = model.inference(data.x, subgraph_loader=loader).to(device)
    model.train()
    Z = Z_all[train_idx]
    sorted_x, sorted_y,sorted_index = sort_dataset_large(Z, torch.argmax(Y,dim=1), num_classes=num_classes)
    prototype = []
    for label in range(num_classes):
        prototype.append(sorted_x[label].mean(axis=0))
    prototype = np.vstack(prototype)

    #compute the cosine similarity to add semantic information
    cos_mat = cosine_similarity(Z_all.detach().cpu(),prototype)
    cos_mat = torch.tensor(np.abs(cos_mat)).to(device) 
    Y_all = torch.zeros(Z_all.shape[0],num_classes).to(device)
    Y_all[train_idx] = Y
    Y_all = args.beta*Y_all + (1-args.beta)*cos_mat

    #propagate the labels to add structural information
    model2 = LabelPropagation(num_layers=args.T, alpha=args.alpha)
    labels =  model2(Y_all, data.edge_index, mask=train_idx)

    return labels

def preprocess(args,data,Y):
    """Preprocess the dataset."""
    num_nodes = data.x.shape[0]
    num_classes = data.y.max().item() + 1   
    Y_all = torch.zeros(num_nodes, num_classes).to(args.device)
    Y = one_hot(Y, num_classes).to(args.device)
    Y_all[data.train_mask] = Y[data.train_mask]
    A = to_dense_adj(data.edge_index)[0].cpu()
    A = A.to(args.device)
    A1 = A.detach().clone()
    int_train_mask = data.train_mask.cpu().numpy().astype(int).reshape(-1,1)
    M = np.matmul(int_train_mask,int_train_mask.T).astype(bool)
    A1 = A1.to('cpu')*M
    A1 = normalize_adj_row(A1.to('cpu'))
    A1 = torch.from_numpy(A1.todense()).to(args.device)
    A = A.to('cpu')
    A = normalize_adj_row(A)
    A = torch.from_numpy(A.todense()).to(args.device)
    for i in range(args.T):
        Y_all = (1-args.alpha)*torch.matmul(A1,Y_all) + args.alpha*Y_all
    L_all = torch.argmax(Y_all,dim=1)
    return L_all,Y_all,A
