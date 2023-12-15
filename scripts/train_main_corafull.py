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
from model import GAT_TYPE
import argparse
from loss import MaximalCodingRateReduction
from evaluate_no_gpu import Linear_classifier
import numpy as np
from utils import setup_logger, get_dataset, setup_seed, preprocess
import os
import logging
from corrupt import default_corrupt


def train(args, model, A, L_all, Y_all, data, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    # compute the loss, see loss.py for details
    loss_func = MaximalCodingRateReduction(
        gam1=args.gam1, gam2=args.gam2, eps=args.eps, corafull=True)
    loss, loss_empi, L_all = loss_func(
        out, data, A, Y_all, args.alpha, args.beta, args.T, data.train_mask)

    total_loss = loss.item()
    loss.backward()
    optimizer.step()
    return total_loss, loss_empi, L_all


@torch.no_grad()
def test(args, model, data, noisy_y, clean_y):
    model.eval()
    out = model(data.x, data.edge_index)

    # evaluate the representation quality, see evaluate.py for details
    noisy_train_labels = noisy_y[data.train_mask]
    train_acc, valid_acc, test_acc = Linear_classifier(
        args, data, out, noisy_train_labels, clean_y)

    return train_acc, valid_acc, test_acc


def get_objs(args):
    device = args.device
    res = dict()
    res['device'] = device
    data = get_dataset(args)
    print(f'num_features: {data.num_features}, num_classes: {data.num_classes}'
          f'num_nodes: {data.num_nodes}, num_edges: {data.num_edges}'
          f'num_train: {data.train_mask.sum()}, num_test: {data.test_mask.sum()}'
          f'num_features: {data.num_features}, num_classes: {data.num_classes}')
    res['data'] = data
    res['model'] = args.type.get_model(data.num_features, args.n_hidden,
                                       args.n_embedding, args.n_layers, args.n_heads,
                                       args.dropout, device, args.use_layer_norm, False, args.use_residual, args.use_resdiual_linear).to(device)
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    # Model settings
    parser.add_argument("--type", dest="type", default=GAT_TYPE.GAT,
                        type=GAT_TYPE.from_string, choices=list(GAT_TYPE))
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='which GPU to use if any (default: cuda:0)')
    parser.add_argument("--n_hidden", dest="n_hidden", default=256, type=int)
    parser.add_argument("--n_embedding", dest="n_embedding",
                        default=512, type=int)
    parser.add_argument("--n_layers", dest="n_layers", default=2, type=int)
    parser.add_argument("--n_heads", dest="n_heads", default=8, type=int)
    parser.add_argument("--dropout", dest="dropout", default=0.5, type=float)
    parser.add_argument("--use_layer_norm",
                        dest="use_layer_norm", default=False, type=bool)
    parser.add_argument("--use_residual", dest="use_residual",
                        default=False, type=bool)
    parser.add_argument("--use_resdiual_linear",
                        dest="use_resdiual_linear", default=False, type=bool)
    # optimizer settings
    parser.add_argument("--lr", dest="lr", default=0.001, type=float)
    parser.add_argument("--wd", dest="wd", default=0.0005, type=float)
    parser.add_argument("--dataset", dest="dataset",
                        default='CoraFull', type=str)
    # ERASE settings
    # hyperparameter setting for reference:
    #                gam1   gam2   eps   beta   alpha   T
    # Corafull        1      2      .01   .7     .7      2
    parser.add_argument("--loss_func", dest="loss_func",
                        default='ERASE', type=str)
    parser.add_argument("--gam1", dest="gam1", default=1, type=float)
    parser.add_argument("--gam2", dest="gam2", default=2, type=float)
    parser.add_argument("--eps", dest="eps", default=0.01, type=float)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--T', type=int, default=2)
    # noise settings
    parser.add_argument('--corrupt_ratio', type=float, default=0.3)
    parser.add_argument('--corrupt_type', type=str, default="asymm")
    # training settings
    parser.add_argument("--seed", dest='seed', default=234, type=int)
    parser.add_argument("--num_epoch", dest='num_epoch', default=400, type=int)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=150)
    # other settings
    parser.add_argument("--exp_dir", dest='exp_dir',
                        default='exp_output', type=str)
    args = parser.parse_args()
    # initialization, set up the file path, logger, tensorboard, etc.
    args.exp_dir = os.path.join(
        args.exp_dir, args.dataset, f'noise_ratio_{args.corrupt_ratio:.1f}')
    os.makedirs(args.exp_dir, exist_ok=True)
    setup_seed(args.seed)
    os.makedirs(os.path.join(args.exp_dir, 'ckpt'), exist_ok=True)
    logger = logging.getLogger(name="train_logger")
    setup_logger(logger, args.exp_dir + '/train_log.txt')
    objs = get_objs(args)
    logger.info(args)
    data = objs['data']
    model = objs['model']
    device = objs['device']
    data = data.to(device)
    model.reset_parameters()
    best_acc_lst = []
    # corrupt the labels, see corrupt.py for details
    clean_y = data.y
    noisy_y = clean_y.clone()
    train_mask = data.train_mask.clone()
    noisy_y[train_mask] = default_corrupt(
        data, args.corrupt_ratio, args.seed, train_mask, t=args.corrupt_type).to(device)
    # preprocess the data to generate the adjacency matrix and the label matrix for training
    L_all, Y_all, A = preprocess(args, data, noisy_y)
    for run in range(args.runs):
        best_acc = 0
        best_val_acc = 0
        best_epoch = 0
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        for epoch in range(1, args.num_epoch+1):
            # train the model
            loss, loss_empi, L_all = train(
                args, model, A, L_all, Y_all, data, optimizer, epoch)
            # test the model
            result = test(args, model, data, L_all, clean_y)
            train_acc, val_acc, test_acc = result

            if val_acc > best_val_acc:
                best_acc = test_acc
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(
                    args.exp_dir, 'ckpt', f'best_model.pth'))
            logger.info(f'Epoch: {epoch:02d}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * val_acc:.2f}%,'
                        f'Test: {100 * test_acc:.2f}%, '
                        f'Loss: {loss:.4f}, ')
            if epoch - best_epoch > args.patience:
                break
        best_acc_lst.append(100*best_acc)
        args.seed = args.seed + 100
        setup_seed(args.seed)
        model.reset_parameters()
    logger.info(
        f'mean:{np.mean(best_acc_lst):.2f} std: {np.std(best_acc_lst):.2f}')


if __name__ == '__main__':
    main()
