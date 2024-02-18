# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/23
 
@Author : Shen Fang
"""
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../../"))

import numpy as np
import torch
import torch.nn as nn
from model_utils import MLP, TemporalConv, TemporalAttn
from typing import Union
from utils import valid_k_d
from graph import DilatedGCN, create_kd_graph, GConv
from basic_ts.utils import load_adj
import torch.nn.functional as F

from .meta import TMeta, SMeta


class LongPatch(nn.Module):
    def __init__(self, num_tokens: int, in_dim: int, out_dim: int):
        super().__init__()
        self.long_patch = nn.Conv2d(in_dim, out_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))
    
    def forward(self, long_term_input: torch.Tensor):
        """
        Squeeze the P long terms patches into 1 patch with TConv.
        
        :param long_term_input: [B, N, P, D]
        :return: [B, N, 1, D]
        """
        long_term_states = long_term_input.transpose(1, 3)  # [B, D, P, N]
        long_term_states = self.long_patch(long_term_states).permute(0, 3, 2, 1)  # [B, N, 1, D]

        return long_term_states

class LRHeaderPeMS03(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len):
        super().__init__()
        self.out = MLP((in_dim * src_len + hid_dim, trg_len * out_dim), norm_type="layer")

        self.long_patch = LongPatch(num_tokens=num_tokens, in_dim=hid_dim, out_dim=hid_dim)
        self.trg_len = trg_len

    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = self.long_patch(long_term_input).squeeze()  # [B, N, D]
        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]
        prediction = self.out(torch.cat([short_term_input, long_term_states], dim=-1))  # [B, N, src_len * in_dim + hid_dim] -> [B, N, trg_len * out_dim]
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, -1]
        return prediction
    
class LRHeader(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len):
        super().__init__()
        self.out = MLP((in_dim * src_len + hid_dim, trg_len * out_dim), act_type="tanh")

        self.long_patch = LongPatch(num_tokens=num_tokens, in_dim=hid_dim, out_dim=hid_dim)
        self.trg_len = trg_len

    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = self.long_patch(long_term_input).squeeze()  # [B, N, D]
        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]
        prediction = self.out(torch.cat([short_term_input, long_term_states], dim=-1))  # [B, N, src_len * in_dim + hid_dim] -> [B, N, trg_len * out_dim]
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, -1]
        return prediction


class LRHeaderNewPeMS03(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len):
        super().__init__()
        self.out = MLP((in_dim * src_len + hid_dim, trg_len * out_dim), act_type="tanh", norm_type="switch1d")

        self.long_patch = LongPatch(num_tokens=num_tokens, in_dim=hid_dim, out_dim=hid_dim)
        self.trg_len = trg_len

    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = self.long_patch(long_term_input).squeeze()  # [B, N, D]
        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]
        prediction = self.out(torch.cat([short_term_input, long_term_states], dim=-1))  # [B, N, src_len * in_dim + hid_dim] -> [B, N, trg_len * out_dim]
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, -1]
        return prediction


class LRLastLongHeader(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len):
        super().__init__()
        self.out = MLP((in_dim * src_len + hid_dim, trg_len * out_dim), act_type="tanh")

        self.trg_len = trg_len
    
    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = long_term_input[:, :, -1]  # [B, N, hid_dim]

        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]

        prediction = self.out(torch.cat([short_term_input, long_term_states], dim=-1))  # [B, N, src_len * in_dim + hid_dim] -> [B, N, trg_len * out_dim]
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, -1]

        return prediction


class LRLastLongHeaderNewPeMS03(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len):
        super().__init__()
        self.out = MLP((in_dim * src_len + hid_dim, trg_len * out_dim), norm_type="switch1d")

        self.trg_len = trg_len
    
    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = long_term_input[:, :, -1]  # [B, N, hid_dim]

        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]

        prediction = self.out(torch.cat([short_term_input, long_term_states], dim=-1))  # [B, N, src_len * in_dim + hid_dim] -> [B, N, trg_len * out_dim]
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, -1]

        return prediction



class LRLastLongHeaderPeMS03(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len):
        super().__init__()
        self.out = MLP((in_dim * src_len + hid_dim, trg_len * out_dim), norm_type="layer")

        self.trg_len = trg_len
    
    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = long_term_input[:, :, -1]  # [B, N, hid_dim]

        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]

        prediction = self.out(torch.cat([short_term_input, long_term_states], dim=-1))  # [B, N, src_len * in_dim + hid_dim] -> [B, N, trg_len * out_dim]
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, L2, N, -1]

        return prediction


class MLPHeader(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len, n_layers):
        super().__init__()
        assert n_layers >= 2, "MLP layer should be bigger than two."

        self.long_patch = LongPatch(num_tokens=num_tokens, in_dim=hid_dim, out_dim=hid_dim)

        dim_list = [in_dim * src_len + hid_dim, out_dim * trg_len]
        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hid_dim)

        self.nn = MLP(dim_list, act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = self.long_patch(long_term_input).squeeze()  # [B, N, D]
        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]

        long_short_states = torch.cat((short_term_input, long_term_states), dim=-1)  # [B, N, src_len * in_dim + hid_dim]

        prediction = self.nn(long_short_states)
        
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, N, trg_len, out_dim] --> [B, trg_len, N, out_dim]
        return prediction  # [B, L, N, C]


class MLPHeaderPeMS03(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len, n_layers):
        super().__init__()
        assert n_layers >= 2, "MLP layer should be bigger than two."

        self.long_patch = LongPatch(num_tokens=num_tokens, in_dim=hid_dim, out_dim=hid_dim)

        dim_list = [in_dim * src_len + hid_dim, out_dim * trg_len]
        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hid_dim)

        self.nn = MLP(dim_list, norm_type="layer")
        self.trg_len = trg_len

    def forward(self, short_term_input: torch.Tensor, long_term_input: torch.Tensor, **kwargs):
        long_term_states = self.long_patch(long_term_input).squeeze()  # [B, N, D]
        B, L, N, _ = short_term_input.size()
        short_term_input = short_term_input.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_input = short_term_input.reshape(B, N, -1)  # [B, N, src_len * in_dim]

        long_short_states = torch.cat((short_term_input, long_term_states), dim=-1)  # [B, N, src_len * in_dim + hid_dim]

        prediction = self.nn(long_short_states)
        
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, N, trg_len, out_dim] --> [B, trg_len, N, out_dim]
        return prediction  # [B, L, N, C]


class MLPLastLongHeader(nn.Module):
    def __init__(self, num_tokens, in_dim, hid_dim, out_dim, src_len, trg_len, n_layers):
        super().__init__()
        assert n_layers >= 2, "MLP layer should be bigger than two."
        
        dim_list = [in_dim * src_len + hid_dim, out_dim * trg_len]

        for i in range(n_layers - 2):
            dim_list.insert(i + 1, hid_dim)

        self.nn = MLP(dim_list, act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_term_history: torch.Tensor, long_term_states: torch.Tensor, **kwargs):
        """
        :param short_term_history, [B, L, N, C]
        :param long_term_states, [B, N, P, D]

        :return prediction: [B, L, N, C]
        """
        B, L, N, _ = short_term_history.size()
        short_term_history = short_term_history.transpose(1, 2)  # [B, N, src_len * in_dim]
        short_term_states = short_term_history.reshape(B, N, -1)  # [B, N, src_len * in_dim]

        long_term_states = long_term_states[:, :, -1]  # [B, N, hid_dim]

        long_short_states = torch.cat((short_term_states, long_term_states), dim=-1)  # [B, N, src_len * in_dim + hid_dim]

        prediction = self.nn(long_short_states)
        
        prediction = prediction.view(B, N, self.trg_len, -1).transpose(1, 2)  # [B, N, trg_len, out_dim] --> [B, trg_len, N, out_dim]
        return prediction  # [B, L, N, C]


class TConvHeader(nn.Module):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int, t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple]):
        super().__init__()
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]
        self.short_nn = MLP((in_dim, hid_dim), act_type="linear")

        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="layer")
        
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)

        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_term_history: torch.Tensor, long_term_states: torch.Tensor, **kwargs):
        """
        :param short_term_history, [B, L, N, C]
        :param long_term_states, [B, N, P, D]

        :return prediction: [B, L, N, C]
        """
        B, N, P, D = long_term_states.size()
        L = short_term_history.size(1)
        long_term_states = long_term_states.transpose(1, 3)  # [B, D, P, N]
        long_term_states = self.long_patch(long_term_states).permute(0, 3, 2, 1)  # [B, N, 1, D]
        short_term_states = self.short_nn(short_term_history).transpose(1, 2)  # [B, N, L, D]
        long_short_states = torch.cat((short_term_states, long_term_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L, D]

        long_short_states = self.t_conv(long_short_states) + long_short_states  # [B, N, L, D]

        prediction = self.out(long_short_states.view(B, N, -1)).view(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


class TAttnHeader(nn.Module):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int, 
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple]):
        super().__init__()
        t_conv_k, t_conv_d = valid_k_d(t_conv_k, t_conv_d)

        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]
        self.long_nn = MLP((hid_dim, trg_len * hid_dim), act_type="linear", norm_type="layer")
        self.short_nn = MLP((in_dim, hid_dim), act_type="linear", norm_type="layer")

        self.conv = nn.ModuleList([TemporalConv(hid_dim, k=t_conv_k[i], d=t_conv_d[i]) for i in range(len(t_conv_k))])
        self.attn = nn.ModuleList([TemporalAttn(hid_dim, hid_dim) for _ in range(len(t_conv_k))]) 
        self.trg_len = trg_len

        self.out = MLP((hid_dim, out_dim), act_type="tanh")

    def forward(self, short_term_history: torch.Tensor, long_term_states: torch.Tensor, **kwargs):
        """
        [summary]
        
        :param short_term_history: [B, src_len, N, C] 
        :param long_term_states: [B, N, P, D]
        :return: prediction, [B, L, N, C]

        long_term_states --> [B, N, 1, D] --> [B, N, trg_len, D]
        """
        B, N, P, D = long_term_states.size()
        long_term_states = long_term_states.transpose(1, 3)  # [B, D, P, N]
        long_term_states = self.long_patch(long_term_states).permute(0, 3, 2, 1).view(B, N, -1)  # [B, N, D]
        t_features = self.long_nn(long_term_states).view(B, N, self.trg_len, -1)  # [B, N, trg_len, D]
        conv_input = t_features

        short_term_states = self.short_nn(short_term_history).transpose(1, 2) # [B, N, src_len, D]
        
        for layer_idx, attn in enumerate(self.attn):
            conv_output = self.conv[layer_idx](conv_input)  # [B, N, trg_len, D] --> # [B, N, trg_len, D]
            conv_output = conv_output.permute(0, 3, 1, 2)
            
            # [B, D, N, trg_len], [B, N, trg_len, src_len]
            conv_output, attention = attn(short_term_states, t_features, conv_output)
            
            conv_input = conv_output.permute(0, 2, 3, 1)  # [B, N, trg_len, D]

        prediction = self.out(conv_input)  # [B, N, trg_len, D]

        return prediction.transpose(1, 2)


class STConvHeader(LongPatch):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int, 
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):
        super().__init__(num_tokens, hid_dim, hid_dim)
        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)


        self.short_nn = MLP((in_dim, hid_dim), act_type="linear")
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="layer")

        # self.s_conv = nn.ModuleList([DilatedGCN(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)

        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_term_history: torch.Tensor, long_term_states: torch.Tensor, **kwargs):
        """
        STConv prediciton header.
        
        :param short_term_history: [B, L1, N, C]
        :param long_term_states:   [B, N, P, D]
        :return: prediction: [B, L2, N, C]
        """
        B, N, P, D = long_term_states.size()
        L = short_term_history.size(1)
        long_term_states = super(STConvHeader, self).forward(long_term_states)  # [B, N, 1, D]
        
        short_term_states = self.short_nn(short_term_history).transpose(1, 2)  # [B, N, L1, D]
        long_short_states = torch.cat((short_term_states, long_term_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]
        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction
    

class LastLongSTConvHeader(nn.Module):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int, 
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):
        super().__init__()
        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)

        self.short_nn = MLP((in_dim, hid_dim), act_type="linear")
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="layer")

        # self.s_conv = nn.ModuleList([DilatedGCN(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)

        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_term_history: torch.Tensor, long_term_states: torch.Tensor, **kwargs):
        """
        STConv prediciton header.
        
        :param short_term_history: [B, L1, N, C]
        :param long_term_states:   [B, N, P, D]
        :return: prediction: [B, L2, N, C]
        """
        B, N, P, D = long_term_states.size()
        L = short_term_history.size(1)
        long_term_states = long_term_states[:, :, [-1]]  # [B, N, 1, D]
        
        short_term_states = self.short_nn(short_term_history).transpose(1, 2)  # [B, N, L1, D]
        long_short_states = torch.cat((short_term_states, long_term_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]
        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


class SAttnHeader(LongPatch):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int):
        super().__init__(num_tokens, hid_dim, hid_dim)
        self.short_nn = MLP((src_len * in_dim, hid_dim), act_type="linear")
        self.out = MLP((hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_term_history: torch.Tensor, long_term_states: torch.Tensor, **kwargs):
        """
        STConv prediciton header.
        
        :param short_term_history: [B, L1, N, C]
        :param long_term_states:   [B, N, P, D]
        :return: prediction: [B, L2, N, C]
        """
        B, N, P, D = long_term_states.size()
        L = short_term_history.size(1)
        long_term_states = super(SAttnHeader,self).forward(long_term_states).squeeze()  # [B, N, 1, D] -> [B, N, D]

        s_attention_map = torch.bmm(long_term_states, long_term_states.transpose(1, 2))  # [B, N, D] * [B, D, N] -> [B, N, N]
        s_attention_map = s_attention_map.softmax(dim=-1)  # [B, N, N]

        short_term_states = self.short_nn(short_term_history.transpose(1, 2).reshape(B, N, -1))  # [B, L1, N, C] -> [B, N, L1, C] -> [B, N, L1*C] -> [B, N, D]

        short_term_states = torch.bmm(s_attention_map, short_term_states)  # [B, N, D]
        
        prediction = self.out(short_term_states).reshape(B, N, self.trg_len, -1).transpose(1, 2)  # [B, N, L2*C] -> [B, N, L2, C] -> [B, L2, N, C]

        return prediction


class SMetaHeader(SMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int):
        super().__init__(hid_dim, in_dim, hid_dim, out_dim)
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]

    def forward(self, short_data: torch.Tensor, long_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
            # graph = np.ones_like(graph) 
        else:
            raise ValueError("Where is my graph file?")
        
        B, N, P, D = long_states.size()
        long_states = long_states.transpose(1, 3)  # [B, D, P, N]
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1).view(B, N, -1)  # [B, N, D]

        dgl_graph = create_kd_graph(1, 1, graph=graph)

        prediction = super().forward(long_states, short_data, dgl_graph[0])

        return F.tanh(prediction)


class TMetaHeader(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int):
        super().__init__(hid_dim, in_dim, hid_dim)
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]

        self.out = MLP((hid_dim*src_len, out_dim*trg_len), act_type="tanh")
        self.trg_len = trg_len

    
    def forward(self, short_data, long_states, **kwargs) -> torch.Tensor:
        B, N, P, D = long_states.size()
        long_states = long_states.transpose(1, 3)  # [B, D, P, N]
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1).view(B, N, -1)  # [B, N, D]

        prediction = super().forward(long_states, short_data)  # [B, L1, N, C]

        prediction = prediction.transpose(1, 2).view(B, N, -1) 
        prediction = self.out(prediction).view(B,N,self.trg_len, -1)  # [B, N, trg_len, out_dim]
        return prediction.transpose(1, 2)


class SConvTMetaHeader(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):
        super().__init__(hid_dim, in_dim, hid_dim)
        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)

        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len

    def forward(self, short_data, long_states, **kwargs) -> torch.Tensor:
        B, N, P, D = long_states.size()
        long_states = long_states.transpose(1, 3)  # [B, D, P, N]
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1).view(B, N, -1)  # [B, N, D]
        
        long_short_states = super().forward(long_states, short_data).transpose(1, 2)  # [B, N, L1, C]
        
        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)
        return prediction



class LastLongTMetaConvSConvHeader(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):
        super().__init__(hid_dim, in_dim, hid_dim)
        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="layer")
        
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)
        
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len
    
    def forward(self, short_data, long_states, **kwargs):
        B, N, P, D = long_states.size()
        L = short_data.size(1)

        long_states = long_states[:, :, [-1]]  # [B, N, 1, D]

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")

        short_states = super().forward(long_states.squeeze(), short_data).transpose(1, 2)  # [B, N, L1, D]

        long_short_states = torch.cat((short_states, long_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


class LastLongTMetaConvSConvHeaderPeMS03(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):
        super().__init__(hid_dim, in_dim, hid_dim)
        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        
        self.long_short = MLP((2 * hid_dim, hid_dim), norm_type="layer")
        
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)
        
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), norm_type="layer")
        self.trg_len = trg_len
    
    def forward(self, short_data, long_states, **kwargs):
        B, N, P, D = long_states.size()
        L = short_data.size(1)

        long_states = long_states[:, :, [-1]]  # [B, N, 1, D]

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")

        short_states = super().forward(long_states.squeeze(), short_data).transpose(1, 2)  # [B, N, L1, D]

        long_short_states = torch.cat((short_states, long_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction



class TMetaConvSConvHeaderNewPeMS03(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):    
        super().__init__(hid_dim, in_dim, hid_dim)
        
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]

        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="switch2d")
        
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)
        
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh", norm_type="switch1d")
        self.trg_len = trg_len
        
    def forward(self, short_data, long_states, **kwargs):
        
        B, N, P, D = long_states.size()
        L = short_data.size(1)

        long_states = long_states.transpose(1, 3)  # [B, D, P, N]  
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1)  # [B, N, 1, D] 

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        short_states = super().forward(long_states.squeeze(), short_data).transpose(1, 2)  # [B, N, L1, D]

        long_short_states = torch.cat((short_states, long_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


class TMetaConvSConvHeader(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):    
        super().__init__(hid_dim, in_dim, hid_dim)
        
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]

        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="layer")
        
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)
        
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len
        
    def forward(self, short_data, long_states, **kwargs):
        
        B, N, P, D = long_states.size()
        L = short_data.size(1)

        long_states = long_states.transpose(1, 3)  # [B, D, P, N]  
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1)  # [B, N, 1, D] 

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        short_states = super().forward(long_states.squeeze(), short_data).transpose(1, 2)  # [B, N, L1, D]

        long_short_states = torch.cat((short_states, long_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


class TMetaConvSConvHeaderP1(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):    
        super().__init__(hid_dim, in_dim, hid_dim)
        
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]

        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="layer")
        
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)
        
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="tanh")
        self.trg_len = trg_len
        
    def forward(self, short_data, long_states, **kwargs):
        
        B, N, P, D = long_states.size()
        L = short_data.size(1)

        long_states = long_states.transpose(1, 3)  # [B, D, P, N]  
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1)  # [B, N, 1, D] 

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        short_states = super().forward(long_states.squeeze(), short_data).transpose(1, 2)  # [B, N, L1, D]

        long_short_states = torch.cat((short_states, long_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


class TMetaConvSConvHeaderPeMS03(TMeta):
    def __init__(self, num_tokens: int, in_dim: int, hid_dim: int, out_dim: int, src_len: int, trg_len: int,
                 t_conv_k: Union[int, tuple], t_conv_d: Union[int, tuple], s_conv_k: Union[int, tuple], s_conv_d: Union[int, tuple]):    
        super().__init__(hid_dim, in_dim, hid_dim)
        
        self.long_patch = nn.Conv2d(hid_dim, hid_dim, kernel_size=(num_tokens, 1), stride=(num_tokens, 1))  # [B, N, P, D]

        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        
        self.long_short = MLP((2 * hid_dim, hid_dim), act_type="linear", norm_type="switch2d")
        
        self.s_conv = nn.ModuleList([GConv(hid_dim, hid_dim, s_k) for s_k in self.s_conv_k])
        self.t_conv = TemporalConv(hid_dim, t_conv_k, t_conv_d)
        
        self.out = MLP((src_len * hid_dim, trg_len * out_dim), act_type="linear", norm_type="switch1d")
        self.trg_len = trg_len
        
    def forward(self, short_data, long_states, **kwargs):
        
        B, N, P, D = long_states.size()
        L = short_data.size(1)

        long_states = long_states.transpose(1, 3)  # [B, D, P, N]  
        long_states = self.long_patch(long_states).permute(0, 3, 2, 1)  # [B, N, 1, D] 

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")
        
        short_states = super().forward(long_states.squeeze(), short_data).transpose(1, 2)  # [B, N, L1, D]

        long_short_states = torch.cat((short_states, long_states.repeat(1, 1, L, 1)), dim=-1)  # [B, N, L1, 2*D]

        long_short_states = self.long_short(long_short_states)  # [B, N, L1, D]
        
        for i, s_conv in enumerate(self.s_conv):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            long_short_states = s_conv(long_short_states, dgl_graph_i)
        
        long_short_states = self.t_conv(long_short_states)  # [B, N, L1, D]

        prediction = self.out(long_short_states.reshape(B, N, -1)).reshape(B, N, self.trg_len, -1).transpose(1, 2)

        return prediction


if __name__ == "__main__":
    adj_mx_list, adj_mx = load_adj(file_path="/home/fangshen/data/MySubway/adj_mx.pkl",  adj_type="original")
    print(adj_mx.shape)
    print(adj_mx_list)
