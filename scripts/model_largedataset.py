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
from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops,remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter_add
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch_sparse import SparseTensor, set_diag

from tqdm import tqdm
class GAT(torch.nn.Module):
    def __init__(self, base_layer, in_channels, hidden_channels, out_channels, num_layers, num_heads,
                 dropout, device, use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear):
        super(GAT, self).__init__()

        self.layers = torch.nn.ModuleList()
        kwargs = {
            'bias':True
        }
        if base_layer is GAT2Conv:
            kwargs['share_weights'] = False
        self.use_adj_norm = use_adj_norm
        self.layers.append(base_layer(in_channels, hidden_channels // num_heads, num_heads, **kwargs))
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_resdiual_linear = use_resdiual_linear
        self.layer_norms = torch.nn.ModuleList()
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        self.residuals = torch.nn.ModuleList()
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(in_channels, hidden_channels))
        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.layers.append(
                base_layer(hidden_channels, hidden_channels // num_heads, num_heads, **kwargs))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
            if use_resdiual_linear and use_residual:
                self.residuals.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(base_layer(hidden_channels, out_channels, 1, **kwargs ))
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.device = device
        self.non_linearity = F.relu
        self.non_linearity_end = F.relu
        print(f"learnable_params: {sum(p.numel() for p in list(self.parameters()) if p.requires_grad)}")

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.layer_norms:
            layer.reset_parameters()
        for layer in self.residuals:
            layer.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
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
        for i, (edge_index, _, size) in enumerate(adjs):

            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_adj_norm:
                edge_index,edge_weight = self.norm(edge_index, x.shape[0], edge_weight, improved=False, dtype=x.dtype)
                new_x = self.layers[i](x,edge_index,edge_attr=edge_weight)
            else:

                new_x = self.layers[i](x,edge_index)

            if 0 < i < self.num_layers - 1 and self.use_residual:
                x = new_x + x
            else:
                x = new_x
            if i < self.num_layers - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
            if i != self.num_layers - 1:
                x = self.non_linearity(new_x)
            x = F.normalize(x,p=1)   
        x = F.normalize(x)
        x = self.non_linearity_end(x)
        return x

    def inference(self, x, subgraph_loader,edge_weight=None):
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
            x_target = x_source[:size[1]]  # Target nodes are always placed first.
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
        elif self is GAT_TYPE.GAT2:
            return "GAT2"
        return "NA"

    def get_model(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,  use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear):
        if self is GAT_TYPE.GAT:
            return GAT(GATConv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,  use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear)
        elif self is GAT_TYPE.GAT2:
            return GAT(GAT2Conv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,  use_layer_norm, use_adj_norm, use_residual, use_resdiual_linear)

    def get_base_layer(self):
        if self is GAT_TYPE.GAT:
            return GATConv
        elif self is GAT_TYPE.GAT2:
            return GAT2Conv
    
class GAT2Conv(MessagePassing):
    r"""
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the layer will share weights.
        (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 share_weights: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GAT2Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, bias=bias)
            self.lin_r = Linear(in_channels[1], heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight=None,
                size: Size = None, return_attention_weights=None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size,edge_weight = edge_weight)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor, 
                size_i: Optional[int],edge_weight=None) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if edge_weight is not None:
            alpha = alpha * edge_weight
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)