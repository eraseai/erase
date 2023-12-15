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
from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from tqdm import tqdm
class GAT(torch.nn.Module):
    def __init__(self, base_layer, in_channels, hidden_channels, out_channels, num_layers, num_heads,
                 dropout, device, use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear):
        """
        initialize the GAT model.

        Args:
            base_layer: The base layer of GAT model.
            in_channels: The input node feature dimension.
            hidden_channels: The hidden node feature dimension.
            out_channels: The output node feature dimension.
            num_layers: The number of layers.
            num_heads: The number of heads.
            dropout: The dropout probability.
            device: The device where the model is running on.
            use_layer_norm: If set to :obj:`True`, will use layer normalization.
            use_adj_norm: If set to :obj:`True`, will normalize the adjacency matrix.
            use_residual: If set to :obj:`True`, will use residual connection.
            use_resdiual_linear: If set to :obj:`True`, will use residual linear connection.
        
        """

        super(GAT, self).__init__()

        #initialize the parameters
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_resdiual_linear = use_resdiual_linear
        kwargs = {'bias':True}
        self.use_adj_norm = use_adj_norm

        #initialize the layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(base_layer(in_channels, hidden_channels // num_heads, num_heads, **kwargs))

        #initialize the layer norm
        self.layer_norms = torch.nn.ModuleList()
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        #initialize the residuals
        self.residuals = torch.nn.ModuleList()
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(in_channels, hidden_channels))
        
        #add the hidden layers
        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.layers.append(
                base_layer(hidden_channels, hidden_channels // num_heads, num_heads, **kwargs))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
            if use_resdiual_linear and use_residual:
                self.residuals.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(base_layer(hidden_channels, out_channels, 1, **kwargs ))

        #initialize the activation function
        self.dropout = dropout
        self.device = device
        self.non_linearity = F.relu
        self.non_linearity_end = F.relu

        print(f"learnable_params: {sum(p.numel() for p in list(self.parameters()) if p.requires_grad)}")

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.layer_norms:
            layer.reset_parameters()
        for layer in self.residuals:
            layer.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        """
        Normalize the adjacency matrix.
        
        Args:
            edge_index: The edge indices.
            num_nodes: The number of nodes.
            edge_weight: The edge weights.
            improved: If set to :obj:`True`, the output will be scaled with
                :math:`1 / \sqrt{\mathrm{deg}(i) \cdot \mathrm{deg}(j)}`.
            dtype: The desired data type of returned tensor.

        Returns:
            edge_index: The normalized edge indices.
            edge_weight: The normalized edge weights. 
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    def forward(self,x,adjs,edge_weight=None):
        """
        the forward process of GAT model.

        Args:
            x: The input node features.
            adjs: The adjacency matrices.
            edge_weight: The edge weights.
        
        Returns:
            x: The output node representations.
        """
        for i, (edge_index, _, size) in enumerate(adjs):

            #dropout process
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            #normalize the adjacency matrix
            if self.use_adj_norm:
                edge_index,edge_weight = self.norm(edge_index, x.shape[0], edge_weight, improved=False, dtype=x.dtype)
                new_x = self.layers[i](x,edge_index,edge_attr=edge_weight)
            else:
                new_x = self.layers[i](x,edge_index)

            #residual process
            if 0 < i < self.num_layers - 1 and self.use_residual:
                x = new_x + x
            else:
                x = new_x

            #layer norm process
            if i < self.num_layers - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
            
            #activation function
            if i != self.num_layers - 1:
                x = self.non_linearity(new_x)
            x = F.normalize(x,p=1)   
        x = F.normalize(x)
        x = self.non_linearity_end(x)
        return x

    def inference(self, x, subgraph_loader,edge_weight=None):
        """
        Inference with the GAT model.

        Args:
            x: The input node features.
            subgraph_loader: The subgraph data loader.
            edge_weight: The edge weights.
        
        Returns:
            The output node representations.
        """
        pbar = tqdm(total=x.size(0) * len(self.layers), leave=False, desc="Layer", disable=False)
        pbar.set_description('Evaluating')
        for i, layer in enumerate(self.layers[:-1]):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x_source = x[n_id].to(self.device)
                x_target = x_source[:size[1]]
                if self.use_adj_norm:
                    edge_index,edge_weight = self.norm(edge_index, x.shape[0], edge_weight, improved=False, dtype=x.dtype)
                    new_x = self.layers[i]((x_source,x_target),edge_index,edge_attr=edge_weight)
                else:
                    new_x = self.layers[i]((x_source,x_target),edge_index)
                if 0 <= i < self.num_layers - 1: 
                    x_target = new_x
                    x_target = self.non_linearity(x_target)

                x_target = F.normalize(x_target,p=1) 
                xs.append(x_target.cpu())  
                pbar.update(batch_size)
            x = torch.cat(xs, dim=0)
        xs=[]
        cnt=0
        for batch_size, n_id, adj in subgraph_loader:
            cnt+=1
            edge_index, _, size = adj.to(self.device)
            x_source = x[n_id].to(self.device)
            x_target = x_source[:size[1]]
            new_x = self.layers[-1]((x_source, x_target), edge_index)
            xs.append(new_x.cpu())
            pbar.update(batch_size)
        x = torch.cat(xs, dim=0)
        pbar.close()
        x = F.normalize(x)
        x = self.non_linearity_end(x)
        return x
    
    
class GAT_TYPE(Enum):
    GAT = auto()
    DPGAT = auto()
    GAT2 = auto()

    @staticmethod
    def from_string(s):
        try:
            return GAT_TYPE[s]
        except KeyError:
            raise ValueError()
    
    def __str__(self):
        if self is GAT_TYPE.GAT:
            return "GAT"
        return "NA"

    def get_model(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,  use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear):
        """Return the GAT model."""
        if self is GAT_TYPE.GAT:
            return GAT(GATConv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,  use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear)
        
    def get_base_layer(self):
        """Return the base layer of GAT model"""
        if self is GAT_TYPE.GAT:
            return GATConv
