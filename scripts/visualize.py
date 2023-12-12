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
import argparse
import torch
from utils import sort_dataset,get_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
from model import GAT_TYPE

@torch.no_grad()
def visualize(data,model,save_path='',num_classes=7):
    model.eval()
    heatmap_path = os.path.join(save_path,f'heatmap.png')

    #get the learned representations
    X = model(data.x,data.edge_index).cpu().numpy()

    #sort the dataset according to the ground truth labels
    sampled_x,_ = sort_dataset(X,data.y,num_classes,stack=True)

    #compute the similarity matrix
    sim = cosine_similarity(sampled_x,sampled_x)

    #visualization, see corresponding function for details
    plot_heatmap(sim,heatmap_path)
    tsne_visualize(X,data.y.cpu(),save_path,num_classes)
    PCA_visualize(X,data.y.cpu(),save_path,num_classes)
    PCA_between_classes(X,data.y.cpu().numpy(),save_path,num_classes)

    
def plot_heatmap(X,save_path):
    """ Plot a heatmap of the similarity matrix. """
    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(X, cmap='Blues')
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    fig.savefig(save_path)

    plt.clf()
    plt.close()
    
    
def tsne_visualize(X,Y,save_path='',num_classes=7):
    """ Plot the t-SNE visualization of the learned representations. """
    save_path = os.path.join(save_path,f'tsne.png')
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    plt.scatter(tsne_results[:,0],tsne_results[:,1],c=Y,cmap='tab10',s=5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.xticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    plt.yticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    plt.clf()
    plt.close()
    
    
def PCA_visualize(X,Y,save_path='',num_classes=7):
    """ Plot the PCA visualization of the learned representations. """
    save_path = os.path.join(save_path,f'pca.png')
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    plt.scatter(pca_results[:,0],pca_results[:,1],c=Y,cmap='tab10',s=5)
    plt.xticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    plt.yticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    ax.set_ylim(-0.1,0.15)
    ax.set_xlim(-0.06,0.20)
    fig.tight_layout()
    fig.savefig(save_path)
    
    plt.clf()
    plt.close()    
    
    
def PCA_between_classes(X,Y,save_path='',num_classes=7):
    """ Plot the PCA visualization between any two classes of the learned representations. """
    cmaps = plt.get_cmap('tab10',num_classes)
    colors = cmaps(np.linspace(0,1,num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i < j:
                X_temp = X[(Y==i)|(Y==j)]
                Y_temp = Y[(Y==i)|(Y==j)]
                new_colors = np.array([cmaps.colors[i],cmaps.colors[j]])
                new_cmap = LinearSegmentedColormap.from_list('custom_cmap', new_colors, N=2)
                save_path_temp = os.path.join(save_path,f'class_{i}{j}.png')
                pca_results = PCA(n_components=2).fit_transform(X_temp)
                fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
                plt.scatter(pca_results[:,0],pca_results[:,1],c=Y_temp,cmap=new_cmap,s=5)
                plt.title(f'Class {i} and Class {j}',fontsize=15,fontname="Times New Roman",fontweight='bold')
                plt.xticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
                plt.yticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
                fig.tight_layout()
                fig.savefig(save_path_temp)
                plt.clf()
                plt.close()    
            

def main():
    parser = argparse.ArgumentParser()
    #settings of the model used for visualization, should be the same as the setting used for training
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument("--type", dest="type", default=GAT_TYPE.GAT, type=GAT_TYPE.from_string, choices=list(GAT_TYPE))
    parser.add_argument('--device', type=str, default='cuda:0', help='which GPU to use if any (default: cuda:0)')
    parser.add_argument("--n_hidden", dest="n_hidden", default=256, type=int)
    parser.add_argument("--n_embedding", dest="n_embedding", default=512, type=int)
    parser.add_argument("--n_layers", dest="n_layers", default=2, type=int)
    parser.add_argument("--n_heads", dest="n_heads", default=8, type=int)
    parser.add_argument("--dropout", dest="dropout", default=0.5, type=float)
    parser.add_argument("--use_layer_norm", dest="use_layer_norm", default=False, type=bool)
    parser.add_argument("--use_residual", dest="use_residual", default=False, type=bool)
    parser.add_argument("--use_resdiual_linear", dest="use_resdiual_linear", default=False, type=bool)
    parser.add_argument("--resume",default='exp_output\\Cora\\asymm_noise_ratio_0.3\ckpt\\best_model.pth')
    parser.add_argument("--corrupt_ratio", dest="corrupt_ratio",default=0.3,type=float)
    parser.add_argument('--corrupt_type',default='asymm',choices=['asymm','symm'])
    args = parser.parse_args()
    data = get_dataset(args)
    device = args.device
    data = data.to(device)
    model = args.type.get_model(data.num_features, args.n_hidden,
                            args.n_embedding, args.n_layers, args.n_heads,
                            args.dropout, device, args.use_layer_norm,False,args.use_residual, args.use_resdiual_linear).to(device)
    
    #load the trained model
    model.load_state_dict(torch.load(args.resume))
    save_path = os.path.join('imgs',args.dataset,f'{args.corrupt_type}_noise_ratio_{args.corrupt_ratio:.1f}')
    os.makedirs(save_path,exist_ok=True)

    #visualize the learned representations
    visualize(data,model,save_path,num_classes=data.num_classes)

if __name__ == '__main__':
    main()

