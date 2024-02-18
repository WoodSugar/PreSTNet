# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/28
 
@Author : Shen Fang
"""

import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_utils import MLP

from typing import Union
from dgl.nn import GATConv, SGConv, ChebConv, APPNPConv


def create_kd_graph(k_th: int, dilation: int, graph: Union[np.array, np.ndarray]) -> list:
    """
    Return the k-th nearest neighbours with dilation.

    :param k_th: int
    :param dilation: int
    :param graph: [N, N].
    :return:
        result_graph: List [ K * Tuple([src_node_id], [trg_node_id]) ].
    Note:
        the graph structure must be self-looped !
    """
    N = graph.shape[0]
    result_graph = []

    base_graph = graph

    base_d_graph = np.linalg.matrix_power(base_graph, dilation)

    sub_graph = np.eye(N)

    for k in range(k_th):
        pos_mask = graph > 0
        neg_mask = sub_graph > 0

        mask = (pos_mask ^ neg_mask) * pos_mask

        src, dst = np.where(mask == True)
        result_graph.append((list(src), list(dst)))

        graph = np.dot(graph, base_d_graph)
        sub_graph = np.dot(sub_graph, base_d_graph)

    return result_graph



class GCN(nn.Module):
    def __init__(self, attn):
        super().__init__()

        self.attn = attn

    def forward(self, k_th_graph: tuple, input_features: torch.Tensor):
        """
        give graph info and features, create the dgl graph and compute the attentional GCN.
        
        :param param: [description]
        :return: [description]
        """
        device = input_features.device

        dgl_graph = dgl.graph(k_th_graph, device=device)
        dgl_graph.ndata["features"] = input_features

        dgl_graph.update_all(self.message_edge, self.message_reduce)

        states = dgl_graph.ndata.pop("new_states")

        return states

    def message_edge(self, edge):
        features = torch.cat((edge.src["features"], edge.dst["features"]), dim=-1)  # [num_edges, B, T, 2 * in_c]

        attention = self.attn(features)

        return {"attention": attention, "features": edge.src["features"]}

    def message_reduce(self, node):
        feature = node.mailbox["features"]
        attention = node.mailbox["attention"]

        attention = attention.softmax(dim=1)
        # attention = F.softmax(attention, dim=1)
        new_state = torch.sum(attention * feature, dim=1)
        del node.mailbox["attention"]
        
        return {"new_states": new_state}


class DilatedGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_kernel: int):
        super().__init__()
        
        self.mlp = MLP((in_dim, out_dim), act_type="linear")
        self.attn = MLP((2 * out_dim, out_dim), act_type="linear")

        self.dilated_conv = nn.ModuleList([GCN(self.attn) for _ in range(num_kernel)])

        self.weight = nn.Parameter(torch.zeros(size=(num_kernel, 1)))

        self.residual = nn.Identity() if in_dim == out_dim else MLP((in_dim, out_dim), act_type="linear", bias=False)

    def init_weight(self):
        nn.init.normal_(self.weight, mean=0, std=0.05)
    

    def forward(self, input_feature: torch.Tensor, kd_graph: list):
        """
        Dilated GCN with attention kernel computing between pair-wise nodes.
        
        :param input_feature: [B, N, T, C]
        :param kd_graph: List[Tuple(List[src_id], List[dst_id])]
        :return: result: [B, N, T, D]
        """
        features = self.mlp(input_feature).transpose(0, 1)  # [N, B, T, D]
        result = 0.

        for i, conv in enumerate(self.dilated_conv):
            result += self.weight[i] * conv(kd_graph[i], features)  # [N, B, T, D]
        
        return F.leaky_relu(result.transpose(0, 1)) + self.residual(input_feature)
            
    

# GATConv
class GConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_kernel, num_gat_heads=2):
        super(GConv, self).__init__()
        
        self.dilated_conv = nn.ModuleList([GATConv(in_dim, out_dim, num_gat_heads) for _ in range(num_kernel)])
        self.weight = nn.Parameter(torch.zeros(size=(num_kernel, 1)))
        
        self.merge = MLP((num_gat_heads*out_dim, out_dim), act_type="linear")

        self.residual = nn.Identity() if in_dim == out_dim else MLP((in_dim, out_dim), act_type="linear", bias=False)

        self.act = nn.LeakyReLU()
        self.k = num_kernel
        self.num_gat_heads = num_gat_heads
        self.out_dim = out_dim
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.weight, mean=0, std=0.05)

    def forward(self, input_feature: torch.Tensor, kd_graph: list):
        """
        DGL GATConv with attention kernel computing between pair-wise nodes.
        
        :param input_feature: [B, N, T, C]
        :param kd_graph: List[Tuple(List[src_id], List[dst_id])]
        :return: result: [B, N, T, D]
        """
        device = input_feature.device
        B, N, T, C = input_feature.size()
        input_feature = input_feature.transpose(0, 1)  # [N, B, T, C]
        result = 0.
        
        for i, conv in enumerate(self.dilated_conv):
            dgl_graph = dgl.graph(kd_graph[i], device=device)
            dgl_graph = dgl.add_self_loop(dgl_graph)
            result += self.weight[i] * conv(dgl_graph, input_feature)    # [N, B, T, num_heads, D]
        
        result = self.merge(result.view(N, B, T, self.num_gat_heads * self.out_dim))  # [N, B, T, D] 
        output = F.leaky_relu(result) + self.residual(input_feature)
        return output.transpose(0, 1)


if __name__ == "__main__":
    k = 2
    d = 1
    graph = np.array([[1, 1, 0, 1, 0], 
                      [1, 1, 1, 0, 0], 
                      [0, 1, 1, 0, 0], 
                      [1, 0, 0, 1, 1], 
                      [0, 0, 0, 1, 1]])
    kd_graph = create_kd_graph(k, d ,graph)
    # print(kd_graph)

    gcn = DilatedGCN(in_dim=2, out_dim=7, num_kernel=2)
    input_data = torch.rand(16, 5, 6, 2)

    out_data = gcn(input_data, kd_graph)
    print(out_data.size())