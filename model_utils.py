# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/11

@Author : Shen Fang
"""
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch.nn.common_types import _size_2_t
from utils import valid_k_d


class MetaLearner(nn.Module):
    def __init__(self, meta_in_c, meta_W_in, meta_W_out):
        super(MetaLearner, self).__init__()
        self.meta_W = nn.Linear(meta_in_c, meta_W_in * meta_W_out, bias=True)
        self.meta_b = nn.Linear(meta_in_c, meta_W_out, bias=True)
        self.W_in = meta_W_in
        self.W_out = meta_W_out

    def forward(self, meta_knowledge):
        """
        :param node_mk: [*X*, meta_in_c], meta knowledge, *X* means the previous dimension.
        :return:
            meta_W: [*X*, meta_W_in, meta_W_out].
            meta_b: [*X*, meta_W_out].
        """
        meta_W = self.meta_W(meta_knowledge)
        meta_b = self.meta_b(meta_knowledge)

        if meta_knowledge.dim() == 3:
            B, X, meta_in_C = meta_knowledge.size()

            meta_W = meta_W.view(B, X, self.W_in, self.W_out)
            meta_b = meta_b.view(B, X, self.W_out)

        elif meta_knowledge.dim() == 2:
            X, meta_in_C = meta_knowledge.size()

            meta_W = meta_W.view(X, self.W_in, self.W_out)
            meta_b = meta_b.view(X, self.W_out)
        elif meta_knowledge.dim() == 4:
            E, B, X, meta_in_C = meta_knowledge.size()

            meta_W = meta_W.view(E, B, X, self.W_in, self.W_out)
            meta_b = meta_b.view(E, B, X, self.W_out)

        return meta_W, meta_b


class MLP(Seq):
    def __init__(self, channels: (list, tuple), act_type: (str, bool) = "leaky_relu", norm_type=None, bias=True):
        mlp = []
        for i in range(1, len(channels)):
            mlp.append(nn.Linear(channels[i- 1], channels[i], bias))
            if act_type:
                mlp.append(act_layer(act_type))

        if norm_type:
            mlp.append(norm_layer(norm_type, channels[-1]))
        super(MLP, self).__init__(*mlp)


class CausalConv1d(nn.Conv1d):
    """
    Class of 1d convolution on temporal axis.

    :param in_c: number of channels of the input data, int.
    :param out_c: number of channels of the output data, int.
    :param k: kernel size, int.
    :param d: dilation rate, int.
    :param stride: stride length of convolution, int.
    :param group: number of groups in the convolution, int.
    :param bias: whether to add the bias in the convolution, bool.
    """

    def __init__(self, in_c, out_c, k, d, stride=1, group=1, bias=True):
        self.padding = (k - 1) * d
        super(CausalConv1d, self).__init__(in_c, out_c, kernel_size=k, stride=stride, padding=self.padding,
                                           dilation=d, groups=group, bias=bias)

    def forward(self, inputs):
        """
        :param inputs: input data, [B, C, T]
        :return: convolution result, [B, D, T]
        """

        result = super(CausalConv1d, self).forward(inputs)  # [B, D, T]

        padding = self.padding[0]
        if padding != 0:
            return result[:, :, :-padding]


        return result


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_c, out_c, k, d, stride=1, group=1, bias=True):
        self.padding = (0, (k - 1) * d)
        super(CausalConv2d, self).__init__(in_c, out_c, kernel_size=(1, k), stride=stride, padding=self.padding,
                                           dilation=(1, d), groups=group, bias=bias)

    def forward(self, inputs):
        """
        :param inputs:  input data, [B, C, N, T].
        :return:  convolution result, [B, D, N, T].
        """

        result = super(CausalConv2d, self).forward(inputs)  # [B, D, N, T1]
        padding = self.padding[1]

        if padding != 0:
            return result[:, :, :, :-padding]

        return result


class TemporalConv(nn.Module):
    def __init__(self, in_c, k, d):
        super().__init__()
        k, d = valid_k_d(k, d)

        self.conv = nn.ModuleList([nn.Conv2d(in_c, 2 * in_c, (1, k[i]), padding=(0, d[i] * (k[i] - 1) // 2),
                                             dilation=(1, d[i])) for i in range(len(d))])
        
    def forward(self, input_data: torch.Tensor):
        # [B, N, T, C]
        conv_input = input_data.permute(0, 3, 1, 2)  # [B, C, N, T]

        for conv_layer in self.conv:
            conv_output = F.glu(conv_layer(conv_input), dim=1) + conv_input
            conv_input = conv_output

        conv_output = conv_input.permute(0, 2, 3, 1)

        return conv_output  # [B, N, T, C]
        

class TemporalAttn(nn.Module):
    def __init__(self, in_c: int, hid_c: int):
        super().__init__()
        self.attn_in2hid = MLP((in_c, hid_c))

    def forward(self, encoder_output:torch.Tensor, input_data:torch.Tensor, input_conved:torch.Tensor):
        """
        :param encoder_output: [B, N, src_len, D].
        :param input_data:  [B, N, trg_len, C].
        :param input_conved:  [B, D, N, trg_len]. 

        :return:
        attention_combine: [B, D, N, trg_len]
        attention: [B, N, trg_len, src_len]
        """
        input_embedded = self.attn_in2hid(input_data)  # [B, N, trg_len, D]
        input_combined = (input_conved.permute(0, 2, 3, 1) + input_embedded) * 0.5  # [B, N, trg_len, D]

        energy = torch.matmul(input_combined, encoder_output.permute(0, 1, 3, 2))  # [B, N, trg_len, src_len]
        attention = F.softmax(energy, dim=-1)  # [B, N, trg_len, src_len]
        attention_encoding = torch.matmul(attention, encoder_output)  # [B, N, trg_len, D]
        attention_combine = (input_embedded + attention_encoding) * 0.5

        return attention_combine.permute(0, 3, 1, 2), attention


def norm_layer(norm_type, nc):
    """
    A universal choice for the normalization layer.

    :param norm_type: normalization type, str.
    :param nc: number of channel dimension, int.

    :return:
        layer: the torch.nn modules.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch1d':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm_type == 'batch2d':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    elif norm_type == "switch1d":
        layer = SwitchNorm1d(nc, using_bn=True)
    elif norm_type == "switch2d":
        layer = SwitchNorm2d(nc, using_bn=True)
    elif norm_type == "layer":
        layer = nn.LayerNorm(nc)
    elif norm_type == "linear":
        layer = nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def act_layer(act_type:str, inplace=True, neg_slope=0.01, n_prelu=1):
    act_type = act_type.lower()

    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leaky_relu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'p_relu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == "sigmoid":
        layer = nn.Sigmoid()
    elif act_type == "tanh":
        layer = nn.Tanh()
    elif act_type == "linear":
        layer = nn.Identity()
    else:
        raise NotImplementedError('activation layer {:s} is not found'.format(act_type))

    return layer


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.05)
            # nn.init.kaiming_normal_(param.data)
        else:
            # nn.init.constant_(param.data, 0.01)
            nn.init.normal_(param.data, mean=0, std=0.05)


def query_kth_neighbour(k_th, dilation, graph):
    """
    Return the k-th nearest neighbours with dilation.

    :param k_th: int
    :param dilation: int
    :param graph: [N, N].
    :return:
        result_graph: [k, N, N].
    Note:
        the graph structure must be self-looped !
    """
    N = graph.size(0)
    result_graph = torch.zeros([k_th, N, N], device=graph.device)

    base_graph = graph

    base_d_graph = torch.matrix_power(base_graph, dilation)

    sub_graph = torch.eye(N, device=graph.device)

    for k in range(k_th):
        pos_mask = graph > 0
        neg_mask = sub_graph > 0

        mask = (pos_mask ^ neg_mask) * pos_mask

        graph_idx = torch.zeros_like(graph).masked_fill(mask, 1.)

        result_graph[k] = graph_idx

        graph = torch.mm(graph, base_d_graph)
        sub_graph = torch.mm(sub_graph, base_d_graph)

    return result_graph


def valid_k_d(k, d):
    k_int = isinstance(k, int)
    d_int = isinstance(d, int)

    n_k = 1 if k_int else len(k)
    n_d = 1 if d_int else len(d)

    n_max, n_min = max(n_k, n_d), min(n_k, n_d)

    if n_max == n_min:
        return [k] if k_int else k, [d] if d_int else d
    else:
        if n_min == 1:
            if n_k == 1:
                base_k = k if k_int else k[0]
                return [base_k for _ in range(n_max)], d
            if n_d == 1:
                base_d = d if d_int else d[0]
                return k, [base_d for _ in range(n_max)]
        else:
            raise ValueError("Length of kernel and dilation rate is not equal")


class SwitchNorm1d(nn.Module):
    """
    Class of switchable normalization of 1 dimensional data (B, N, C).

    :param in_channels: number of channels of the input data, int.
    :param eps: a small number to avoid the situation of divided by zeros, float.
    :param momentum: the momentum adopted in batch normalization, float (0, 1).
    :param using_moving_average: whether to using momentum in batch normalization, bool.
    :param using_bn: whether to use batch normalization, bool.
    :param last_gamma: whether to use the linear mapping of last output, bool.
    """

    def __init__(self, in_channels, eps=1e-5, momentum=0.997,
                 using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma

        self.f = nn.Softmax(0)

        self.weight = nn.Parameter(torch.ones(1, 1, in_channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, in_channels))

        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))

        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, in_channels, 1))
            self.register_buffer('running_var', torch.zeros(1, in_channels, 1))

        self.reset_parameters()

    @staticmethod
    def _check_input_dim(inputs):
        """
        Check the input data dimension.

        :param inputs: input data, [B, N, C]
        """

        if inputs.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(inputs.dim()))

    def reset_parameters(self):
        """
        Reset the parameters, including the running mean and var in the bn block,
        the weight and bias of the last linear mapping.
        """

        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()

        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, inputs):
        """
        :param inputs: input data, [B, N, C].
        :return: normalized output data, [B, N, C].
        """

        SwitchNorm1d._check_input_dim(inputs)
        inputs = inputs.transpose(1, 2)  # [B, C, N]

        mean_in = inputs.mean(-1, keepdim=True)  # [B, C, 1]
        var_in = inputs.var(-1, keepdim=True)  # [B, C, 1]

        mean_ln = mean_in.mean(1, keepdim=True)  # [B, 1, 1]
        temp = var_in + mean_in ** 2  # [B, C, 1]
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2  # [B, 1, 1]

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)  # [1, C, 1]
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2  # [1, C, 1]

                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)  # [1, C, 1]
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)  # [1, C, 1]

                    self.running_var.mul_(self.momentum)  # [1, C, 1]
                    self.running_var.add_((1 - self.momentum) * var_bn.data)  # [1, C, 1]
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)

            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        mean_weight = self.f(self.mean_weight)
        var_weight = self.f(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn  # [B, C, 1]
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn  # [B, C, 1]
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln  # [B, C, 1]
            var = var_weight[0] * var_in + var_weight[1] * var_ln  # [B, C, 1]

        inputs = (inputs - mean) / (var + self.eps).sqrt()  # [B, C, N]
        inputs = inputs.transpose(1, 2)  # [B, N, C]

        return self.weight * inputs + self.bias


class SwitchNorm2d(nn.Module):
    """
    Class of switchable normalization of 2 dimensional data (B, N, T, C).

    :param in_channels: number of channels of the input data, int.
    :param eps: a small number to avoid the situation of divided by zeros, float.
    :param momentum: the momentum adopted in batch normalization, float (0, 1).
    :param using_moving_average: whether to using momentum in batch normalization, bool.
    :param using_bn: whether to use batch normalization, bool.
    :param last_gamma: whether to use the linear mapping of last output, bool.
    """

    def __init__(self, in_channels, eps=1e-5, momentum=0.997,
                 using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, 1, 1, in_channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, in_channels))

        self.f = nn.Softmax(0)

        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))

        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, in_channels, 1))
            self.register_buffer('running_var', torch.zeros(1, in_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters, including the running mean and var in the bn block,
        the weight and bias of the last linear mapping.
        """

        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    @staticmethod
    def _check_input_dim(inputs):
        """
        Check the input data dimension.

        :param inputs: input data, [B, N, T, C]
        """

        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(inputs.dim()))

    def forward(self, inputs):
        """
        :param inputs: input data, [B, N, T, C].
        :return: normalized output data, [B, N, T, C].
        """

        self._check_input_dim(inputs)
        B, N, T, C = inputs.size()
        inputs = inputs.permute(0, 3, 1, 2).view(B, C, -1)  # [B, C, N*T]

        mean_in = inputs.mean(-1, keepdim=True)  # [B, C, 1]
        var_in = inputs.var(-1, keepdim=True)  # [B, C, 1]

        mean_ln = mean_in.mean(1, keepdim=True)  # [B, 1, 1]
        temp = var_in + mean_in ** 2  # [B, C, 1]
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2  # [B, 1, 1]

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)  # [1, C, 1]
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2  # [1, C, 1]
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        mean_weight = self.f(self.mean_weight)
        var_weight = self.f(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        inputs = (inputs - mean) / (var + self.eps).sqrt()
        inputs = inputs.view(B, C, N, T)
        inputs = inputs.permute(0, 2, 3, 1)

        return self.weight * inputs + self.bias


class TemporalEmbedding(nn.Module):
    def __init__(self, temporal_para: dict):
        super(TemporalEmbedding, self).__init__()
        self.embed_holiday = MLP(temporal_para["holiday"])  # [1, 2, 2]
        self.embed_time = MLP(temporal_para["time"])  # [1, 2, 2]
        self.embed_final = MLP(temporal_para["final"])  # [4, 4, 4]

        assert temporal_para["final"][0] == temporal_para["holiday"][-1] + temporal_para["time"][-1]

    def forward(self, temporal_data):
        """
        :param temporal_data: [B, T, C].
        :return:
            embedding_result: [B, T, D].
        """
        holiday_data, time_data = torch.chunk(temporal_data, 2, dim=-1)
        holiday_result = self.embed_holiday(holiday_data)
        time_result = self.embed_time(time_data)

        hybrid_result = self.embed_final(torch.cat((holiday_result, time_result), dim=-1))

        return hybrid_result


class WeatherEmbedding(nn.Module):
    def __init__(self, weather_para: dict):
        super(WeatherEmbedding, self).__init__()
        self.embed_discrete = MLP(weather_para["discrete"])  # [8, 4, 4]
        self.embed_continuous = MLP(weather_para["continuous"])  # [4, 4, 4]
        self.embed_final = MLP(weather_para["final"])  # [8, 8, 8]

        assert weather_para["final"][0] == weather_para["discrete"][-1] + weather_para["continuous"][-1]

    def forward(self, weather_data):
        """
        :param weather_data: [B, T, C].
        :return:
        """
        discrete_data = weather_data[:, :, :8]
        continuous_data = weather_data[:, :, 8:]

        discrete_result = self.embed_discrete(discrete_data)
        continuous_result = self.embed_continuous(continuous_data)

        hybrid_result = self.embed_final(torch.cat((discrete_result, continuous_result), dim=-1))

        return hybrid_result


class PoIEmbedding(nn.Module):
    def __init__(self, poi_para: tuple):
        super(PoIEmbedding, self).__init__()
        self.embed = MLP(poi_para)  # [15, 8, 8]

    def forward(self, poi_data):
        """
        :param poi_data: [B, N, C].
        :return:
            [B, N, D].
        """
        return self.embed(poi_data)


class SpatialConv(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        

    def forward(self):
        pass



if __name__ == '__main__':
    conv = CausalConv2d(in_c=2, out_c=8, k=3, d=1)
    x = torch.randn(32, 2, 278, 6)
    y = conv(x)

    print(y.size())
