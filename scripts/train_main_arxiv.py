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
from tqdm import tqdm
from loss_largedataset import MaximalCodingRateReduction
from evaluate_gpu import Linear_classifier
import torch
import argparse
from model_largedataset import GAT_TYPE
import torch.optim as optim
from torch_geometric.loader import NeighborSampler
import os
import logging
from ogb.nodeproppred import Evaluator,PygNodePropPredDataset
import torch_geometric.transforms as T
import numpy as np
from corrupt_largedataset import default_corrupt
from utils import denoising_propagation,compute_semantic_structral_label,setup_logger,setup_seed


def train(args,model, data,Y_all,loader,optimizer,epoch):
    #initialization
    model.train()
    device = args.device
    total_loss = 0
    discrimn_loss_empi = 0
    compress_loss_empi = 0
    delta_R = 0

    #train the model using batched data, the maximum accuracy is usually achieved within first several iters of the first epoch
    for batch_size, n_id, adjs in tqdm(loader, leave=False, desc=f"Epoch {epoch}", dynamic_ncols=True):
        L = torch.argmax(Y_all,dim=1).reshape(-1,1)
        adjs = [adj.to(device) for adj in adjs]
        x = data.x[n_id].to(device)
        optimizer.zero_grad()
        out = model(x, adjs)
        criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
        loss,loss_empi,loss_theo= criterion(out,L[n_id],args.num_classes)
        discrimn_loss_empi += loss_empi[0]
        compress_loss_empi += loss_empi[1]
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        delta_R+=(loss_theo[0]-loss_theo[1])
    discrimn_loss_empi = discrimn_loss_empi/len(loader)
    compress_loss_empi = compress_loss_empi/len(loader)
    delta_R = delta_R/len(loader)
    total_loss = total_loss/len(loader)

    return discrimn_loss_empi, compress_loss_empi, total_loss,delta_R
        
@torch.no_grad()
def test(model, data,clean_y,L, loader, split_idx,evaluator):
    model.eval()
    #test the model, see evaluate_gpu.py for details
    with torch.no_grad():
        out = model.inference(data.x, subgraph_loader=loader)
        out = out.detach()
        noisy_train_labels = L[split_idx['train']]
        train_acc,valid_acc,test_acc = Linear_classifier(out, split_idx,noisy_train_labels,clean_y, evaluator)
    return  train_acc,valid_acc, test_acc

def get_objs(args):
    device = args.device
    res = dict()
    res['device'] = device
    transforms = T.ToUndirected()
    dataset = PygNodePropPredDataset(name=args.dataset, transform=transforms,root='dataset')
    res['dataset'] = dataset
    data = dataset[0]
    res['data'] = data
    args.num_classes = data.y.max().item() + 1
    args.n_features = data.x.shape[1]
    res['evaluator'] = Evaluator(name=args.dataset)
    res['split_idx'] = dataset.get_idx_split()
    res['train_idx'] = res['split_idx']['train']
    res['model'] = args.type.get_model(data.num_features, args.n_hidden,
                                args.n_embedding, args.n_layers, args.n_heads,
                                args.dropout, device, args.use_layer_norm,args.use_adj_norm, args.use_residual, args.use_resdiual_linear).to(device)
    return res

def main():
    parser = argparse.ArgumentParser(description='Supervised Learning')
    #Model settings
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument("--type", dest="type", default=GAT_TYPE.GAT, type=GAT_TYPE.from_string, choices=list(GAT_TYPE))
    parser.add_argument('--n_features', type=int, default=256,help='dimension of feature dimension (default: 128)')
    parser.add_argument('--n_hidden', type=int, default=64,help='dimension of hidden dimension (default: 256)')
    parser.add_argument('--n_embedding', type=int, default=512,help='dimension of embedding dimension (default: 512)')
    parser.add_argument('--n_classes', type=int, default=40,help='number of classes (default: 40)')
    parser.add_argument('--n_heads', type=int, default=8,help='number of heads for attention module (default: 8)')
    parser.add_argument('--n_layers', type=int, default=2,help='number of layers of model (default: 2)')
    parser.add_argument('--use_layer_norm', action='store_true', default=False)
    parser.add_argument('--use_residual', action='store_true', default=False)
    parser.add_argument('--use_resdiual_linear', action='store_true', default=False)
    parser.add_argument('--use_adj_norm',action='store_true', default=False)
    #optimizer settings
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',help='dataset for training (default: ogbn-arxiv)')
    parser.add_argument('--epochs', type=int, default=5,help='number of epochs for training (default: 20)')
    parser.add_argument('--batch_size', type=int, default=200,help='input batch size for training (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=5e-4,help='weight decay (default: 5e-4)')
    #ERASE settings
    #hyperparameter setting for reference:
    #                gam1   gam2   eps   beta   alpha   T
    #ogbn-arxiv      1      2      .05   .5     .6      50
    parser.add_argument("--gam1", dest="gam1", default=1, type=float)
    parser.add_argument("--gam2", dest="gam2", default=2, type=float)
    parser.add_argument('--eps', type=float, default=0.05,help='eps squared (default: 0.05)')
    parser.add_argument('--alpha',default=0.6,type=float)
    parser.add_argument('--beta',default=0.5,type=float)
    parser.add_argument('--T',default=50,type=int)
    #corrupt settings
    parser.add_argument('--corrupt_type',default='asymm',choices=['asymm','symm'])
    parser.add_argument("--corrupt_ratio", dest="corrupt_ratio",default=0.5,type=float)
    #training settings
    parser.add_argument('--exp_dir', type=str, default='exp_output',help='base directory for saving information. (default: '' )')
    parser.add_argument('--device', type=str, default='cuda:0',help='which GPU to use if any (default: cuda:0)')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate (default: 0.5)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)  
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--mertic', type=str, default='acc')
    parser.add_argument('--patience',default=15,type=int)
    args = parser.parse_args()
    #set up logger, seed, and file paths
    setup_seed(args.seed)
    args.exp_dir = os.path.join(args.exp_dir,args.dataset,f'{args.corrupt_type}_noise_ratio_{args.corrupt_ratio:.1f}')
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'ckpt'), exist_ok=True)
    logger = logging.getLogger(name="train_logger")
    setup_logger(logger, args.exp_dir + '/train_log.txt')
    objs = get_objs(args)
    data = objs['data']
    split_idx = objs['split_idx']
    train_idx = objs['train_idx']
    model = objs['model']
    evaluator = objs['evaluator']
    device = objs['device']
    best_acc_lst =[]
    #set up data loader
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=args.n_layers * [5],batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],batch_size=4096, shuffle=False, num_workers=args.num_workers)
    data = data.to(device)
    train_idx = split_idx['train']
    #corrupt the labels
    clean_y = data.y
    noisy_y = clean_y.clone()
    noisy_y[train_idx] = default_corrupt(data,train_idx,args.corrupt_ratio,args.seed,args.corrupt_type).to(device).reshape(-1,1)
    logger.info(args)
    #propagate the noisy labels
    denoised_y= denoising_propagation(args,noisy_y)
    run = 0
    while run < args.runs:
        model.reset_parameters()
        optimizer = optim.Adam(list(model.parameters()),lr=args.lr, weight_decay=args.wd)
        iter=0
        best_acc=0
        best_val_acc=0
        best_epoch = 0
        for epoch in range(args.epochs):
            #compute robust labels using sematic information
            Y_all = compute_semantic_structral_label(args,data,model,subgraph_loader,denoised_y[train_idx],train_idx)
            #train the model
            loss= train(args,model, data, Y_all,train_loader, optimizer,epoch)
            total_loss = loss[2]
            #evaluate the model
            if epoch % args.eval_steps == 0:
                Y_all = compute_semantic_structral_label(args,data,model,subgraph_loader,denoised_y[train_idx],train_idx)
                L = torch.argmax(Y_all,dim=1).reshape(-1,1)
                train_acc, valid_acc, test_acc = test(model, data,clean_y,L, subgraph_loader, split_idx,evaluator)
                logger.info(
                    f'Run: {run:02d},'
                    f'epoch: {epoch}'
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%, '
                    f'Loss: {total_loss:.2f},')   
                if valid_acc >= best_val_acc:
                    best_acc = test_acc  
                    best_val_acc = valid_acc
                    best_epoch = epoch 
                    torch.save(model.state_dict(), os.path.join(args.exp_dir, 'ckpt',f'best_model.pt'))
            if epoch - best_epoch > args.patience:
                break
        best_acc_lst.append(100*best_acc)
        run += 1
    logger.info(f'mean:{np.mean(best_acc_lst):.2f} std: {np.std(best_acc_lst):.2f}')
if __name__ == '__main__':
    main()
