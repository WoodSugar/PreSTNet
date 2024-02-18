import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from model_utils import MLP, TemporalConv, act_layer, norm_layer
from timm.models.vision_transformer import trunc_normal_
from graph import DilatedGCN, create_kd_graph, GConv
from typing import Union
import numpy as np
from utils import valid_k_d


class PatchEmbedding(nn.Module):
    """
    Embed the long term history data into several segments.
    [B, N, C, T=(P*L)] =-> [B, N, D, P]
    T is the very long-term total length. 
    P is the number of segments you want to remain, L is the length of each segment.
    """
    def __init__(self, patch_length, in_channel, embed_dim, norm=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = embed_dim
        self.patch_length = patch_length
        
        self.input_embedding = nn.Conv2d(in_channel, embed_dim, 
                                         kernel_size=(patch_length, 1), 
                                         stride=(patch_length, 1))  # [B, L, N, C]
        
        self.norm_layer = norm_layer(norm, embed_dim) if norm is not None else nn.Identity()
        self.norm = norm
            
        self.act_layer = act_layer("linear")

    def forward(self, long_term_history: torch.Tensor):
        """
        :param long_term_history: [B, N, C, P * L]. P is the number of time segments (patches), and L is the each patch length.
        """
        B, N, C, T = long_term_history.size()
        long_term_history = long_term_history.unsqueeze(-1)  # [B, N, C, P * L, 1]
        long_term_history = long_term_history.reshape(B * N, C, T, 1)  # [B*N, C, P*L, 1] 
        
        output_embedding = self.input_embedding(long_term_history)  # [B*N, D, P, 1]
        
        output_embedding = output_embedding.squeeze(-1).view(B, N, -1, self.out_channel) # [B, N, P, D]
        if self.norm in ["batch2d", "batch1d"]:
            output_embedding = output_embedding.permute(0, 3, 1, 2)  # [B, D, N, P]
            output_embedding = self.norm_layer(output_embedding)     
            output_embedding = output_embedding.permute(0, 2, 3, 1)  # [B, N, P, D]
        else:
            output_embedding = self.norm_layer(output_embedding) 

        output_embedding = self.act_layer(output_embedding)
        output_embedding = output_embedding.transpose(-2, -1) # [B, N, D, P]
        assert output_embedding.size(-1) == (T // self.patch_length), "Patching embedding with wrong computing length."
        return output_embedding
    

class PositionEmbedding(nn.Module):
    """
    Assign the position encoding of the time series (each segment).
    [B, N, P, C] =-> [B, N, P, C]
    The position encoding is randomly generated and can be optimized.
    if index (effected on dimension P) is specified, fetch the position encoding result according to the index,
    otherwise, fetch the position encoding result with the first P numbers.
    """
    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)  # [MAX_LEN, C]
    
    def forward(self, input_data, index=None):
        B, N, P, C = input_data.size()
        input_data = input_data.view(B * N, P, C)  # [B*N, P, C]
        
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)  # [P, C] -> [1, P, C]
        else:
            assert len(index) == P, "index size must be equal to P."
            pe = self.position_embedding[index].unsqueeze(0)  # [len(index), C] -> [1, len(index), C]

        input_data = input_data + pe
        input_data = self.dropout(input_data)
        
        return input_data.view(B, N, P, C)


class PositionSinCos(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=1000):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super().__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, embed_dim) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        
        self.register_buffer('pe', pe)  # pe值是不参加训练的
        self.pe.requires_grad = False
    
    def forward(self, input_data: torch.Tensor, index=None):
        B, N, P, C = input_data.size()
        input_data = input_data.view(B * N, P, C)  # [B*N, P, C]

        if index is None:
            pe = self.pe[:input_data.size(1), :].unsqueeze(0)
        else:
            assert len(index) == P, "index size must be equal to P."
            pe = self.pe[index].unsqueeze(0)  # [len(index), C] -> [1, len(index), C]            
        
        input_data = input_data + pe  # 输入的最终编码 = word_embedding + positional_embedding
        
        input_data = self.dropout(input_data)
        return input_data.view(B, N, P, C)


class ZeroEmbedding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = torch.zeros(max_len, hidden_dim, requires_grad=False)

    def forward(self, input_data: torch.Tensor, index=None):
        B, N, P, C = input_data.size()
        input_data = input_data.view(B * N, P, C)  # [B*N, P, C]
        
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)  # [P, C] -> [1, P, C]
        else:
            assert len(index) == P, "index size must be equal to P."
            pe = self.position_embedding[index].unsqueeze(0)  # [len(index), C] -> [1, len(index), C]

        pe = pe.to(input_data.device)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        
        return input_data.view(B, N, P, C)
    

class RandomEmbedding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = torch.randn(max_len, hidden_dim, requires_grad=False)

    def forward(self, input_data: torch.Tensor, index=None):
        B, N, P, C = input_data.size()
        input_data = input_data.view(B * N, P, C)  # [B*N, P, C]
        
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)  # [P, C] -> [1, P, C]
        else:
            assert len(index) == P, "index size must be equal to P."
            pe = self.position_embedding[index].unsqueeze(0)  # [len(index), C] -> [1, len(index), C]

        pe = pe.to(input_data.device)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        
        return input_data.view(B, N, P, C)


class PosEncoding(nn.Module):
    def __init__(self, num_tokens, embed_dim, method, dropout=0.1, max_len=1000):
        super().__init__()
        assert method in ["sincos", "param", "arange", "zero", "random"]
        self.method = method
        self.num_tokens = num_tokens
        if method == "sincos":
            self.embedding = PositionSinCos(embed_dim, dropout, max_len)
        elif method == "param":
            self.embedding = PositionEmbedding(embed_dim, dropout, max_len)
        elif method == "arange":
            self.embedding = nn.Embedding(num_tokens, embed_dim)
        elif method == "zero":
            self.embedding = ZeroEmbedding(embed_dim, dropout, max_len)
        elif method == "random":
            self.embedding = RandomEmbedding(embed_dim, dropout, max_len)
        else:
            raise ValueError("wrong embedding method")
    
    def forward(self, input_data: torch.Tensor, index=None):
        device = input_data.device
        B, N, P, C = input_data.size()
        if self.method == "arange":
            input_data = input_data.view(B * N, P, C)  # [B*N, P, C]

            pe = torch.arange(0, self.num_tokens).unsqueeze(0).to(device)  # [1, P]
            pe = self.embedding(pe)  # [1, P, D]
            if index is None:
                pe = pe[:, :P]
            else:
                assert len(index) == P, "index size must be equal to P."
                pe = pe[:, index]
            input_data = input_data + pe
            return input_data.view(B, N, P, C)
        else:
            input_data = self.embedding(input_data, index)
            return input_data


class TemporalAttention(nn.Module):
    def __init__(self, in_dim:int, hid_dim:int):
        super().__init__()
        self.attn_in2hid = MLP((in_dim, hid_dim))

    def forward(self, encoder_output: torch.Tensor, input_data: torch.Tensor, input_conved: torch.Tensor):
        """
        :param encoder_output: [B, N, T1, D].
        :param input_data:  [B, N, T2, C].
        :param input_conved:  [B, N, T2, D].
        :return:
            attention_combine: [B, N, T2, D]
            attention: [B, N, T2, T1]
        """
        # [B, N, T2, D]
        input_embedded = self.attn_in2hid(input_data)
        input_combined = (input_conved + input_embedded) * 0.5
        # [B, N, T2, D] * [B, N, D, T1] -> [B, N, T2, T1]
        energy = torch.matmul(input_combined, encoder_output.permute(0, 1, 3, 2))
        attention = F.softmax(energy, dim=-1) # [B, N, T2, T1(Norm)]
        # [B, N, T2, T1(Norm)] * [B, N, T1, D] -> [B, N, T2, D]
        attention_encoding = torch.matmul(attention, encoder_output)

        attention_combine = (attention_encoding  + input_embedded) * 0.5
        return attention_combine, attention


class MaskGenerator(nn.Module):
    """
    given a time series with length [num_tokens], randomly generate several masks with a [mask_ration].
    """
    def __init__(self, num_tokens, mask_ratio) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        masked_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:masked_len]
        self.unmasked_tokens = mask[masked_len:]
        if self.sort:
            self.masked_tokens.sort()
            self.unmasked_tokens.sort()
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        # B, N, P, C = input_data.size()
        # input_data = input_data.view(B * N, P, C)  # [B*N, P, C]
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class TransformerLayers(nn.Module):
    """
    A multi-layer transformer encoder
    [B, N, T, C] =-> [B, N, T, C]
    """
    def __init__(self, hid_dim, n_layers, mlp_ration, num_heads=4, dropout=0.1, act=F.relu):
        super().__init__()
        self.d_model = hid_dim
        encoder_layers = nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim*mlp_ration, dropout,
                                                    activation=act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def forward(self, src: torch.Tensor):
        B, N, T, C = src.size()
        src = src * math.sqrt(self.d_model)  # avoid the effect of x / sqrt(d), so first product with sqrt(d).
        src = src.view(B * N, T, C)  # [B*N, T, C]
        src = src.permute(1, 0, 2)  # [T, B*N, C]
        output = self.transformer_encoder(src, mask=None)
        output = output.permute(1, 0, 2).view(B, N, T, C)  # [B, N, T, C]
        return output


class STTransformerLayers(nn.Module):
    def __init__(self, hid_dim, n_layers, mlp_ration, num_heads=4, dropout=0.1, act=F.relu):
        super().__init__()
        self.d_model = hid_dim
        encoder_layers = nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim*mlp_ration, dropout, activation=act, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
    
    def forward(self, src: torch.Tensor):
        B, N, T, C = src.size()
        src = src * math.sqrt(self.d_model)
        output = torch.cat([self.transformer_encoder(src[:, i]).unsqueeze(0) for i in range(N)], dim=0)   # [N, B, T, C]
        output = output.transpose(0, 1)  # [B, N, T, C]
        return output


class ST_TransformerLayers(nn.Module):
    def __init__(self, hid_dim, n_layers, mlp_ration, num_heads=4, dropout=0.1, act=F.relu):
        super().__init__()
        self.d_model = hid_dim
        self.t_encoder = nn.ModuleList([nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim*mlp_ration, dropout, activation=act) for _ in range(n_layers)])
        self.s_encoder = nn.ModuleList([nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim*mlp_ration, dropout, activation=act) for _ in range(n_layers)])

        self.n_layers = n_layers
    def forward(self, src: torch.Tensor):
        B, N, T, C = src.size()
        src = src * math.sqrt(self.d_model)

        output = src
        for i in range(self.n_layers):
            output = output.reshape(B*N, T, C).permute(1, 0, 2)  # [T, B*N, C]

            output = self.t_encoder[i](output)
            output = output.permute(1, 0, 2).view(B, N, T, C).transpose(1, 2)  # [B, T, N, C]
            output = output.reshape(B*T, N, -1).permute(1, 0, 2)  # [N, B*T, C]

            output = self.s_encoder[i](output)  # [N, B*T, C]
            output = output.permute(1, 0, 2).view(B, T, N, C).transpose(1, 2)  # [B, N, T, C]
        
        return output


class ST_ParallelTransformerLayers(nn.Module):
    def __init__(self, hid_dim, n_layers, mlp_ration, num_heads=4, dropout=0.1, act=F.relu):
        super().__init__()
        self.d_model = hid_dim
        en_t_layers = nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim*mlp_ration, dropout, activation=act)
        en_s_layers = nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim*mlp_ration, dropout, activation=act)

        self.t_layers = nn.TransformerEncoder(en_t_layers, n_layers)
        self.s_layers = nn.TransformerEncoder(en_s_layers, n_layers)

        self.merge = MLP((2 * hid_dim, hid_dim), act_type="linear")

    def forward(self, src: torch.Tensor):
        B, N, T, C = src.size()
        src = src * math.sqrt(self.d_model)
        t_input = src.view(B * N, T, C)  # [B*N, T, C]
        t_input = t_input.permute(1, 0, 2)  # [T, B*N, C]

        s_input = src.transpose(1, 2).reshape(B * T, N, C)  # [B*T, N, C]
        s_input = s_input.permute(1, 0, 2)  # [N, B*T, C]

        t_output = self.t_layers(t_input, mask=None).permute(1, 0, 2).view(B, N, T, C)  # [B, N, T, C]
        s_output = self.s_layers(s_input, mask=None).permute(1, 0, 2).view(B, T, N, C).transpose(1, 2)  # [B, N, T, C]

        st_output = torch.cat([t_output, s_output], dim=-1)
        st_output = self.merge(st_output)
        
        return st_output


class SConv_TTransformerLayers(TransformerLayers):
    def __init__(self, hid_dim, n_layers, mlp_ration, num_t_heads=4, dropout=0.1, act=F.relu, s_conv_k=1, s_conv_d=1, num_s_heads=2):
        super().__init__(hid_dim, n_layers, mlp_ration, num_t_heads, dropout, act)

        self.s_conv_k, self.s_conv_d = valid_k_d(s_conv_k, s_conv_d)
        self.gcn = nn.ModuleList([GConv(hid_dim, hid_dim, s_k, num_s_heads) for s_k in self.s_conv_k])

    def forward(self, src: torch.Tensor, graph: Union[np.array, np.ndarray]=None):
        t_output = super().forward(src)  # [B, N, T, C]
        s_output = t_output
        for i, s_conv in enumerate(self.gcn):
            dgl_graph_i = create_kd_graph(self.s_conv_k[i], self.s_conv_d[i], graph)
            s_output = s_conv(s_output, dgl_graph_i)

        return s_output 


# ============================================================================================================
# 目前最常用的Encoder，只有时序Transformer
class BaseTAEncoder(nn.Module):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, n_layers, num_heads, mlp_ration, en_dropout, en_norm):
        super().__init__()
        self.patch_length = patch_length
        self.num_tokens = num_tokens
        self.patch_embedding = PatchEmbedding(patch_length, in_channel, embed_dim, patch_embed_norm)
        self.posE = PosEncoding(num_tokens, embed_dim, posE_method, posE_dropout)
        self.posE_method = posE_method

        self.mask = MaskGenerator(num_tokens, mask_ration)
        self.selected_feature = list(range(in_channel))

        self.compute = TransformerLayers(embed_dim, n_layers, mlp_ration, num_heads, en_dropout)
        self.encoder_norm = norm_layer(en_norm, embed_dim)
        self.En_norm = en_norm
        self.embed_dim = embed_dim

        self.init_weights()

    def init_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)

    def forward(self, long_term_history: torch.Tensor, mask=True, graph: Union[np.array, np.ndarray]=None):
        """:param long_term_history, [B, N, C, T=(P * L)]
        :Return
            if mask:
                hidden_states_unmasked: torch.Tensor, [B, N, P_unmask, C] or [B, N, P, C]
                unmasked_token_idx:     list with length P_unmask
                masked_token_idx:       list with length P_mask
        """
        B, N = long_term_history.size(0), long_term_history.size(1)  # [B, N, C, T=(P*L)]
        patches = self.patch_embedding(long_term_history)  # [B, N, C, P]
        patches = patches.permute(0, 1, 3, 2) # [B, N, P, C]
        # position encoding
        patches = self.posE(patches)
        
        if mask:
            unmask_token_idx, mask_token_idx =self.mask()
            encoder_input = patches[:, :, unmask_token_idx]
        else:
            unmask_token_idx, mask_token_idx = None, None
            encoder_input = patches

        # encoding
        if graph is None:
            hidden_states_unmask = self.compute(encoder_input)
        else:
            hidden_states_unmask = self.compute(encoder_input, graph)

        if self.En_norm in ["batch1d", "batch2d"]:
            hidden_states_unmask = hidden_states_unmask.permute(0, 3, 1, 2)  
            hidden_states_unmask = self.encoder_norm(hidden_states_unmask)
            hidden_states_unmask = hidden_states_unmask.permute(0, 2, 3, 1)  
        else:
            hidden_states_unmask = self.encoder_norm(hidden_states_unmask).view(B, N, -1, self.embed_dim)

        return hidden_states_unmask, unmask_token_idx, mask_token_idx

    def get_reconstructed_masked_tokens(self, reconstruction_full:torch.Tensor, 
                                            real_value_full: torch.Tensor,
                                            unmasked_token_idx: list, masked_token_idx: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the reconstructed masked tokens and the corresponding real value.

        param: reconstruction_full, [B, N, P, L*C]. 
        param: real_value_full, [B, N, C, P * L].

        return: 
            reconstruction_masked_tokens, # [B, N, P_mask, L, C]
            label_masked_tokens,          # [B, N, P_mask, L, C]
        """
        B, N = reconstruction_full.size(0), reconstruction_full.size(1)
        # get the reconstructed masked tokens
        
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_idx):, :]  # [B, N, P_mask, L*C]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, len(masked_token_idx), self.patch_length, -1)  # [B, N, P_mask, L, C]
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2) # [B, P_mask * L * C, N]

        # real_value_full [B, N, C, L*P] -> [B, L*P, N, C] -> [B, P, N, C, L]
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(dimension=1, size=self.patch_length, step=self.patch_length)

        # [B, P, N, C, L] -> [B, P, N, C, L] -> [B, N, P, L, C]
        label_full = label_full[:, :, :, self.selected_feature, :].permute(0, 2, 1, 4, 3) # [B, N, P, L, C]

        label_masked_tokens = label_full[:, :, masked_token_idx, :].contiguous()  # [B, N, P_mask, L, C]

        # label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)  # [B, P_mask * L * C, N]
        
        return reconstruction_masked_tokens, label_masked_tokens


# 基于常用BaseTAEncoder的改进形式，将前向计算改进为单独针对时间的显式循环Attention，速度慢，整体效果不如BaseTAEncoder
class SingleTAEncoder(BaseTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, 
                         num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm)

        self.compute = STTransformerLayers(embed_dim, n_layers, mlp_ration, num_heads, En_dropout)
    
    def forward(self, long_term_history: torch.Tensor, mask=True):
        return super().forward(long_term_history, mask)


# 基于常用BaseTAEncoder的改进形式，前向计算改为层叠式TAttn+SAttn
class STAStackEncoder(BaseTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, 
                         num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm)

        self.compute = ST_TransformerLayers(embed_dim, n_layers, mlp_ration, num_heads, En_dropout)
    
    def forward(self, long_term_history: torch.Tensor, mask=True):
        return super().forward(long_term_history, mask)


# 基于常用EncoderUnmask的改进形式，前向计算改为并行的TAttn+SAttn
class STAConcatEncoder(BaseTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, 
                         num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm)

        self.compute = ST_ParallelTransformerLayers(embed_dim, n_layers, mlp_ration, num_heads, En_dropout)
    
    def forward(self, long_term_history: torch.Tensor, mask=True):
        return super().forward(long_term_history, mask)


# 基于常用EncoderUnmask的改进形式，前向计算为Tattn + SConv
class SCTAEncoder(BaseTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, n_layers, num_heads, mlp_ration, en_sconv_k, en_sconv_d, en_dropout, en_norm):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                         num_tokens, mask_ration, n_layers, num_heads, mlp_ration, en_dropout, en_norm)

        self.compute = SConv_TTransformerLayers(embed_dim, n_layers, mlp_ration, num_heads, en_dropout, s_conv_k=en_sconv_k, s_conv_d=en_sconv_d)

    def forward(self, long_term_history: torch.Tensor, mask=True, graph: Union[np.array, np.ndarray]=None):
        return super().forward(long_term_history, mask, graph)


# ============================================================================================================
# ============================================================================================================
# 其他改进的Encoder形式
class EncoderFull(nn.Module):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, n_layers, num_heads, mlp_ration, En_dropout, En_norm):
        super().__init__()
        self.patch_length = patch_length
        self.num_tokens = num_tokens
        self.patch_embedding = PatchEmbedding(patch_length, in_channel, embed_dim, patch_embed_norm)
        self.posE = PosEncoding(num_tokens, embed_dim, posE_method, posE_dropout)
        
        self.mask = MaskGenerator(num_tokens, mask_ration)
        
        self.selected_feature = list(range(in_channel))

        self.compute = TransformerLayers(embed_dim, n_layers, mlp_ration, num_heads, En_dropout)
        self.encoder_norm = norm_layer(En_norm, embed_dim)
        self.En_norm = En_norm

    def forward(self, long_term_history: torch.Tensor, mask=True):
        B, N = long_term_history.size(0), long_term_history.size(1)  # [B, N, C, T=(P*L)]
        patches = self.patch_embedding(long_term_history)  # [B, N, C, P]
        patches = patches.permute(0, 1, 3, 2) # [B, N, P, C]

        if mask: 
            unmask_token_idx, mask_token_idx =self.mask()
            unmasked_input = patches[:, :, unmask_token_idx, :]  # [B, N, P_unmask, C]
            mask_random_input = torch.randn(B, N, len(mask_token_idx), patches.size(-1)).to(long_term_history.device)
            patch_idx = unmask_token_idx + mask_token_idx  # e.g. [0, 2, 4, 5] + [1, 3] = [0 2 4 5 1 3] 
            patch_idx = sorted(range(len(patch_idx)), key=lambda k:patch_idx[k])  # [0 4 1 5 2 3]
            encoder_input = torch.cat([unmasked_input, mask_random_input], dim=-2)[:, :, patch_idx]

        else:
            unmask_token_idx, mask_token_idx = None, None
            encoder_input = patches
        
        # position encoding
        encoder_input = self.posE(encoder_input)
        hidden_states_full = self.compute(encoder_input)
        if self.En_norm in ["batch1d", "batch2d"]:
            hidden_states_full = hidden_states_full.permute(0, 3, 1, 2)  
            hidden_states_full = self.encoder_norm(hidden_states_full)
            hidden_states_full = hidden_states_full.permute(0, 2, 3, 1)  
        else:
            hidden_states_full = self.encoder_norm(hidden_states_full)
        return hidden_states_full, unmask_token_idx, mask_token_idx

    def get_reconstructed_masked_tokens(self, reconstruction_full:torch.Tensor, 
                                            real_value_full: torch.Tensor,
                                            unmasked_token_idx: list, masked_token_idx: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the reconstructed masked tokens and the corresponding real value.

        param: reconstruction_full, [B, N, P, L*C]. 
        param: real_value_full, [B, N, C, P * L].

        return: 
            reconstruction_masked_tokens, # [B, N, P_mask, L, C]
            label_masked_tokens,          # [B, N, P_mask, L, C]
        """
        B, N = reconstruction_full.size(0), reconstruction_full.size(1)
        # get the reconstructed masked tokens
        
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_idx):, :]  # [B, N, P_mask, L*C]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, len(masked_token_idx), self.patch_length, -1)  # [B, N, P_mask, L, C]
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2) # [B, P_mask * L * C, N]

        # real_value_full [B, N, C, L*P] -> [B, L*P, N, C] -> [B, P, N, C, L]
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(dimension=1, size=self.patch_length, step=self.patch_length)

        # [B, P, N, C, L] -> [B, P, N, C, L] -> [B, N, P, L, C]
        label_full = label_full[:, :, :, self.selected_feature, :].permute(0, 2, 1, 4, 3) # [B, N, P, L, C]

        label_masked_tokens = label_full[:, :, masked_token_idx, :].contiguous()  # [B, N, P_mask, L, C]

        # label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)  # [B, P_mask * L * C, N]
        
        return reconstruction_masked_tokens, label_masked_tokens


class TConvEncoderMask(nn.Module):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, t_conv_k, t_conv_d, En_norm):
        super().__init__()
        self.patch_length = patch_length
        self.num_tokens = num_tokens
        self.patch_embedding = PatchEmbedding(patch_length, in_channel, embed_dim, patch_embed_norm)
        self.posE = PosEncoding(num_tokens, embed_dim, posE_method, posE_dropout)

        self.mask = MaskGenerator(num_tokens, mask_ration)
        self.selected_feature = list(range(in_channel))

        self.compute = TemporalConv(embed_dim, t_conv_k, t_conv_d)
        self.encoder_norm = norm_layer(En_norm, embed_dim)
        self.En_norm = En_norm

    def forward(self, long_term_history: torch.Tensor, mask=True):
        B, N = long_term_history.size(0), long_term_history.size(1)  # [B, N, C, T=(P*L)]
        patches = self.patch_embedding(long_term_history)  # [B, N, C, P]
        patches = patches.permute(0, 1, 3, 2) # [B, N, P, C]
        
        # position encoding
        patches = self.posE(patches)

        if mask:
            unmask_token_idx, mask_token_idx =self.mask()
            encoder_input = patches[:, :, unmask_token_idx]
        else:
            unmask_token_idx, mask_token_idx = None, None
            encoder_input = patches

        # encoding
        hidden_states_unmask = self.compute(encoder_input)
        if self.En_norm in ["batch1d", "batch2d"]:
            hidden_states_unmask = hidden_states_unmask.permute(0, 3, 1, 2)  
            hidden_states_unmask = self.encoder_norm(hidden_states_unmask)
            hidden_states_unmask = hidden_states_unmask.permute(0, 2, 3, 1)  
        else:
            hidden_states_unmask = self.encoder_norm(hidden_states_unmask)

        return hidden_states_unmask, unmask_token_idx, mask_token_idx

    def get_reconstructed_masked_tokens(self, reconstruction_full:torch.Tensor, 
                                            real_value_full: torch.Tensor,
                                            unmasked_token_idx: list, masked_token_idx: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the reconstructed masked tokens and the corresponding real value.

        param: reconstruction_full, [B, N, P, L*C]. 
        param: real_value_full, [B, N, C, P * L].

        return: 
            reconstruction_masked_tokens, # [B, N, P_mask, L, C]
            label_masked_tokens,          # [B, N, P_mask, L, C]
        """
        B, N = reconstruction_full.size(0), reconstruction_full.size(1)
        # get the reconstructed masked tokens
        
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_idx):, :]  # [B, N, P_mask, L*C]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, len(masked_token_idx), self.patch_length, -1)  # [B, N, P_mask, L, C]
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2) # [B, P_mask * L * C, N]

        # real_value_full [B, N, C, L*P] -> [B, L*P, N, C] -> [B, P, N, C, L]
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(dimension=1, size=self.patch_length, step=self.patch_length)

        # [B, P, N, C, L] -> [B, P, N, C, L] -> [B, N, P, L, C]
        label_full = label_full[:, :, :, self.selected_feature, :].permute(0, 2, 1, 4, 3) # [B, N, P, L, C]

        label_masked_tokens = label_full[:, :, masked_token_idx, :].contiguous()  # [B, N, P_mask, L, C]

        # label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)  # [B, P_mask * L * C, N]
        
        return reconstruction_masked_tokens, label_masked_tokens


class TConvEncoderFull(nn.Module):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method,
                 num_tokens, mask_ration, t_conv_k, t_conv_d, En_norm):
        super().__init__()
        self.patch_length = patch_length
        self.num_tokens = num_tokens
        self.patch_embedding = PatchEmbedding(patch_length, in_channel, embed_dim, patch_embed_norm)
        self.posE = PosEncoding(num_tokens, embed_dim, posE_method, posE_dropout)
        
        self.mask = MaskGenerator(num_tokens, mask_ration)
        
        self.selected_feature = list(range(in_channel))

        self.compute = TemporalConv(embed_dim, t_conv_k, t_conv_d)
        self.encoder_norm = norm_layer(En_norm, embed_dim)
        self.En_norm = En_norm

    def forward(self, long_term_history: torch.Tensor, mask=True):
        B, N = long_term_history.size(0), long_term_history.size(1)  # [B, N, C, T=(P*L)]
        patches = self.patch_embedding(long_term_history)  # [B, N, C, P]
        patches = patches.permute(0, 1, 3, 2) # [B, N, P, C]

        if mask:
            unmask_token_idx, mask_token_idx =self.mask()
            unmasked_input = patches[:, :, unmask_token_idx, :]  # [B, N, P_unmask, C]
            mask_random_input = torch.randn(B, N, len(mask_token_idx), patches.size(-1)).to(long_term_history.device)
            patch_idx = unmask_token_idx + mask_token_idx  # e.g. [0, 2, 4, 5] + [1, 3] = [0 2 4 5 1 3] 
            patch_idx = sorted(range(len(patch_idx)), key=lambda k:patch_idx[k])  # [0 4 1 5 2 3]
            encoder_input = torch.cat([unmasked_input, mask_random_input], dim=-2)[:, :, patch_idx]

        else:
            unmask_token_idx, mask_token_idx = None, None
            encoder_input = patches

        # position encoding
        encoder_input = self.posE(encoder_input)

        hidden_states_full = self.compute(encoder_input)
        if self.En_norm in ["batch1d", "batch2d"]:
            hidden_states_full = hidden_states_full.permute(0, 3, 1, 2)  
            hidden_states_full = self.encoder_norm(hidden_states_full)
            hidden_states_full = hidden_states_full.permute(0, 2, 3, 1)  
        else:
            hidden_states_full = self.encoder_norm(hidden_states_full)
        return hidden_states_full, unmask_token_idx, mask_token_idx
    
    def get_reconstructed_masked_tokens(self, reconstruction_full:torch.Tensor, 
                                            real_value_full: torch.Tensor,
                                            unmasked_token_idx: list, masked_token_idx: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the reconstructed masked tokens and the corresponding real value.

        param: reconstruction_full, [B, N, P, L*C]. 
        param: real_value_full, [B, N, C, P * L].

        return: 
            reconstruction_masked_tokens, # [B, N, P_mask, L, C]
            label_masked_tokens,          # [B, N, P_mask, L, C]
        """
        B, N = reconstruction_full.size(0), reconstruction_full.size(1)
        # get the reconstructed masked tokens
        
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_idx):, :]  # [B, N, P_mask, L*C]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, len(masked_token_idx), self.patch_length, -1)  # [B, N, P_mask, L, C]
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2) # [B, P_mask * L * C, N]

        # real_value_full [B, N, C, L*P] -> [B, L*P, N, C] -> [B, P, N, C, L]
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(dimension=1, size=self.patch_length, step=self.patch_length)

        # [B, P, N, C, L] -> [B, P, N, C, L] -> [B, N, P, L, C]
        label_full = label_full[:, :, :, self.selected_feature, :].permute(0, 2, 1, 4, 3) # [B, N, P, L, C]

        label_masked_tokens = label_full[:, :, masked_token_idx, :].contiguous()  # [B, N, P_mask, L, C]

        # label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)  # [B, P_mask * L * C, N]
        
        return reconstruction_masked_tokens, label_masked_tokens


# ============================================================================================================
# ============================================================================================================
# 单层TADecoder
class BaseTADecoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int, mlp_ration: int, num_heads:int, norm_method:str, dropout:float=0.1):
        super().__init__()

        self.norm_method = norm_method
        self.compute = TransformerLayers(hid_dim, n_layers, mlp_ration, num_heads, dropout)
        self.decoder_norm = norm_layer(norm_method, hid_dim)
        self.out_layer = MLP((hid_dim, out_dim), act_type="linear")  # out_dim = patch_length * in_channel

    def forward(self, input_data: torch.Tensor):
        output = self.compute(input_data)
        if self.norm_method in ["batch2d", "batch1d"]:
            output = output.permute(0, 3, 1, 2)  
            output = self.decoder_norm(output)  
            output = output.permute(0, 2, 3, 1)  
        else:
            output = self.decoder_norm(output)  
        output = self.out_layer(output)
        return output
        
# 基于MLP的Decoder
class LinearDecoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int, mlp_ration: int, num_heads:int, dropout:float=0.1):
        super().__init__()
        channels = [hid_dim for _ in range(n_layers)]
        self.compute = MLP(channels, act_type="linear")

    def forward(self, input_data: torch.Tensor):
        output = self.compute(input_data)
        return output
        

# 一个比较奇怪的TA+TConvDecoder，似乎在MetaMSNet与AutoMSNet中扮演过Decoder
class TConvTADecoder(nn.Module):
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, 
                 t_conv_k:int, n_layers:int, num_tokens:int):
        super().__init__()
        self.in2hid = MLP((in_dim, hid_dim))
        self.hid2in = MLP((hid_dim, out_dim))

        self.pos_embedding = nn.Embedding(num_tokens, in_dim)
        self.conv = nn.ModuleList([TemporalConv(hid_dim, t_conv_k, 1) for _ in range(n_layers)])

        self.attn = TemporalAttention(in_dim, hid_dim)

        self.num_tokens = num_tokens

    def forward(self, input_data: torch.Tensor):
        B, N, T, C = input_data.size()
        device = input_data.device
        # [B, N, T]
        pos = torch.arange(0, self.num_tokens).unsqueeze(0).unsqueeze(0).repeat(B, N, 1).to(device)
        input_data = input_data + self.pos_embedding(pos) # [B, N, T, Embed]

        conv_input = self.in2hid(input_data)  # [B, N, T, D]

        for conv_layer in self.conv:
            conv_output = conv_layer(conv_input)
            conv_output, attention = self.attn(conv_input, input_data, conv_output)
            conv_input = conv_output
        
        conv_output = self.hid2in(conv_input)
        return conv_output