# -*- coding: utf-8 -*-
"""
@Time   : 2023/10/09

@Author : Shen Fang
"""
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from typing import Tuple, Any, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from timm.models.vision_transformer import trunc_normal_

from model_utils import MLP, act_layer, norm_layer, TemporalConv, TemporalAttn
from .module import PatchEmbedding, PositionEmbedding, PositionSinCos, MaskGenerator
from .module import SingleTAEncoder, STAStackEncoder, STAConcatEncoder, SCTAEncoder
from .module import TransformerLayers, BaseTAEncoder, BaseTADecoder, TConvTADecoder, LinearDecoder
from .module import EncoderFull, TConvEncoderFull, TConvEncoderMask


class TemporalLayers(nn.Module):
    def __init__(self, mode: str, in_c: int, k: Union[int , List]=None, d: int=None):
        super().__init__()
        assert mode in ["conv", "attn"], "Error temporal computing mode."

        if mode == "conv":
            assert k is not None, "when temporal computing is set to conv, k and d should be given."
            assert d is not None, "when temporal computing is set to conv, k and d should be given."
            self.layer = TemporalConv(in_c, k, d)
        elif mode == "attn":
            self.layer = TemporalAttn(in_c, in_c)
        self.mode = mode

    def forward(self, encoder_output: torch.Tensor, input_data:torch.Tensor=None, input_conved:torch.Tensor=None):
        if self.mode == "conv":
            output_data = self.layer(encoder_output)
        elif self.mode == "attn":
            output_data = self.layer(encoder_output, input_data, input_conved)
        return output_data


class SpatialLayers(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward():
        pass


class STLayer(nn.Module):
    def __init__(self, hid_dim: int, n_layers: int, 
                 en_t_k: Union[int, List], en_t_d: Union[int, List], 
                 mlp_ration: int=None, num_heads:int=None, dropout:float=0.1):
        super().__init__()

        self.t_compute = nn.ModuleList([TemporalLayers("conv", hid_dim, en_t_k, en_t_d) for _ in range(n_layers)])
        self.s_compute = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        self.out_layer = TemporalLayers("attn", hid_dim) 

        self.n_layers = n_layers

    def forward(self, src: torch.Tensor):
        B, N, T, C = src.size()
        output = src
        for i in range(self.n_layers):
            output = self.t_compute[i](output)
            output = self.s_compute[i](output)

        output, attn_map = self.out_layer(output, output, output.permute(0, 3, 1, 2))
        # [B, D, N, trg_len]
        return output.permute(0, 2, 3, 1)  # [B, N, T, C]


class STFormer(nn.Module):
    def __init__(self, patch_length, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, 
                 mask_ratio, encoder_depth, decoder_depth, mode, **kwargs):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_length = patch_length
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        self.mode = mode
        self.selected_feature = list(range(in_channel))

        # norm layers
        self.encoder_norm = norm_layer("layer", embed_dim)
        self.decoder_norm = norm_layer("layer", embed_dim)

        # encoder part
        # # patching and embedding
        self.patch_embedding = PatchEmbedding(patch_length, in_channel, embed_dim, norm="linear")
        self.pos_embedding = PositionEmbedding(embed_dim, dropout)
        self.pos_encoding = PositionSinCos(embed_dim, dropout)
        # # mask
        self.mask = MaskGenerator(num_token, mask_ratio)
        # encoder
        self.encoder = STLayer(hid_dim=embed_dim, n_layers=encoder_depth, 
                               en_t_k=3, en_t_d=(1, 2, 2))

        # decoder part
        # #feature transformation
        self.e2d_embed = MLP((embed_dim, embed_dim), act_type=None)
        # # masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # decoder
        # self.decoder = TransformerLayers(hid_dim=embed_dim,
        #                                  n_layers=decoder_depth,
        #                                  mlp_ration=mlp_ratio,
        #                                  num_heads=num_heads, dropout=dropout)
        
        self.decoder = TConvTADecoder(embed_dim, embed_dim, embed_dim, 3, 2, self.num_token)
        # # prediction head
        self.output_layer = MLP((embed_dim, patch_length * in_channel), act_type="tanh")

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.pos_embedding.position_embedding, -.1, .1)
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history: torch.Tensor, mask=True):
        """
        Encoding process including patching, pos encoding, add mask, and finally transformer layer.

        :param long_term_history, [B, N, C, T=(P * L)]
        :Return
            if mask:
                hidden_states_unmasked: torch.Tensor, [B, N, P, C]
                unmasked_token_idx:     list with length P_unmask
                masked_token_idx:       list with length P_mask
        """
        B, N = long_term_history.size(0), long_term_history.size(1)  # [B, N, C, T=(P*L)]
        # patching
        patches = self.patch_embedding(long_term_history)  # [B, N, C, P]
        patches = patches.permute(0, 1, 3, 2)  # [B, N, P, C]
        # position encoding
        patches = self.pos_encoding(patches)  # [B, N, P, C]
        # mask
        if mask:
            unmasked_token_idx, masked_token_idx = self.mask()

            unmasked_input = patches[:, :, unmasked_token_idx, :]  # [B, N, P_unmask, C]
            masked_input = self.pos_embedding.position_embedding[masked_token_idx, :].unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # [B, N, P_mask, C]
            patch_idx = unmasked_token_idx + masked_token_idx  # e.g. [0, 2, 4, 5] + [1, 3] = [0 2 4 5 1 3] 
            
            patch_idx = sorted(range(len(patch_idx)), key=lambda k:patch_idx[k])  # [0 4 1 5 2 3]
            encoder_input = torch.cat([unmasked_input, masked_input], dim=-2)[:, :, patch_idx]
        else:
            unmasked_token_idx, masked_token_idx = None, None
            encoder_input = patches
        
        hidden_states = self.encoder(encoder_input)  # [B, N, P, C]
        
        return hidden_states, unmasked_token_idx, masked_token_idx
    
    def decoding(self, hidden_states: torch.Tensor, masked_token_idx: torch.Tensor):
        B, N = hidden_states.size(0), hidden_states.size(1)
        # hidden_states = self.e2d_embed(hidden_states)

        hidden_states = self.decoder(hidden_states)
        reconstruction_full = self.decoder_norm(hidden_states)
        reconstruction_full = self.output_layer(hidden_states)

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full: torch.Tensor, 
                                        real_value_full: torch.Tensor,
                                        unmasked_token_idx: list, masked_token_idx: list) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = reconstruction_full.size(0), reconstruction_full.size(1)
        
        reconstruction_masked_tokens = reconstruction_full[:, :, masked_token_idx,:]  # [B, N, P_mask, L*C] 
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, len(masked_token_idx), self.patch_length, -1)  # [B, N, P_mask, L, C]
        
        # real_value_full [B, N, C, L*P] -> [B, L*P, N, C] -> [B, P, N, C, L]
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(dimension=1, size=self.patch_length, step=self.patch_length)

        # [B, P, N, C, L] -> [B, P, N, C, L] -> [B, N, P, L, C]
        label_full = label_full[:, :, :, self.selected_feature, :].permute(0, 2, 1, 4, 3) # [B, N, P, L, C]

        label_masked_tokens = label_full[:, :, masked_token_idx, :].contiguous()  # [B, N, P_mask, L, C]

        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True):
        """
        :param history_data: [B, T = (L * P), N, C]
        :return:
            if pre-train:
                if train == True:
                    reconstruction_masked_tokens: [B, P_mask * L * C, N]
                    label_masked_tokens:          [B, P_mask * L * C, N]
                else:
                    reconstruction_masked_tokens  [B, N, P_mask, L, C]
                    label_masked_tokens:          [B, N, P_mask, L, C]
            else:
                hidden_states_full: [B, N, P, C]
        """
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
        if self.mode == "pre-train":
            hidden_states, unmasked_token_idx, masked_token_idx = self.encoding(history_data, mask=True)
            # tensor of [B, N, P, C], list with length P_unmask, list with length P_mask

            reconstruction_full = self.decoding(hidden_states, masked_token_idx)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, 
                                                                                                     unmasked_token_idx, masked_token_idx)
            # [B, N, P_mask, L, C]

            if train == True:
                reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
                # [B, N, P_mask, L, C] -> [B, P_mask * L * C, N]
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full  # [B, N, P, C]
            

class BackBone(nn.Module):
    def __init__(self, patch_length, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, 
                 mask_ratio, encoder_depth, decoder_depth, mode, **kwargs):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_length = patch_length
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_tokens = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        self.mode = mode
        self.selected_feature = list(range(in_channel))

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # encoder part
        # # patching and embedding 
        self.patch_embedding = PatchEmbedding(patch_length, in_channel, embed_dim, norm="layer")
        self.pos_encoding =  PositionEmbedding(embed_dim, dropout=dropout)
        # # masking 
        self.mask = MaskGenerator(num_token, mask_ratio)
        # encoder
        self.encoder = TransformerLayers(hid_dim=embed_dim, 
                                         n_layers=encoder_depth, 
                                         mlp_ration=mlp_ratio, 
                                         num_heads=num_heads, dropout=dropout)

        # decoder part
        # # feature transformation
        self.e2d_embed = nn.Linear(embed_dim, embed_dim)
        # # masking 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # decoder
        self.decoder = TransformerLayers(hid_dim=embed_dim,
                                         n_layers=decoder_depth,
                                         mlp_ration=mlp_ratio,
                                         num_heads=num_heads, dropout=dropout)
        
        # # prediction head
        self.output_layer = nn.Linear(embed_dim, patch_length * in_channel)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.pos_encoding.position_embedding, -.02, .02)
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history: torch.Tensor, mask=True):
        """
        Encoding process including patching, pos encoding, add mask, and finally transformer layer.

        :param long_term_history, [B, N, C, T=(P * L)]
        :Return
            if mask:
                hidden_states_unmasked: torch.Tensor, [B, N, P_unmask, C] or [B, N, P, C]
                unmasked_token_idx:     list with length P_unmask
                masked_token_idx:       list with length P_mask
        """
        B, N = long_term_history.size(0), long_term_history.size(1)  # [B, N, C, T=(P*L)]
        # patching
        patches = self.patch_embedding(long_term_history)  # [B, N, C, P]
        patches = patches.permute(0, 1, 3, 2)  # [B, N, P, C]
        # position encoding
        patches = self.pos_encoding(patches)  # [B, N, P, C]
        # masking
        if mask:
            unmasked_token_idx, masked_token_idx = self.mask()
            encoder_input = patches[:, :, unmasked_token_idx, :]  # [B, N, P_unmask, C]
        else:
            unmasked_token_idx, masked_token_idx = None, None
            encoder_input = patches

        # encoding
        hidden_states_unmasked = self.encoder(encoder_input)  # [B, N, P_unmask, C]
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(B, N, -1, self.embed_dim)  # [B, N, P_unmask, C]

        return hidden_states_unmasked, unmasked_token_idx, masked_token_idx
        
    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: list) -> torch.Tensor:
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)
        hidden_states_unmasked = self.e2d_embed(hidden_states_unmasked)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.pos_encoding(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P_unmask+P_mask, C])

        hidden_states_full = self.decoder(hidden_states_full)  # [B, N, P_unmask + P_mask, C])
        hidden_states_full = self.decoder_norm(hidden_states_full)
        
        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full)  # [B, N, P, C] -> [B, N, P, L * in_C]
        return reconstruction_full

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

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        """
        :param history_data: [B, T = (L * P), N, C]
        :return:
            if pre-train:
                if train == True:
                    reconstruction_masked_tokens: [B, P_mask * L * C, N]
                    label_masked_tokens:          [B, P_mask * L * C, N]
                else:
                    reconstruction_masked_tokens  [B, N, P_mask, L, C]
                    label_masked_tokens:          [B, N, P_mask, L, C]
            else:
                hidden_states_full: [B, N, P, C]
        """
        
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
        if self.mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = self.encoding(history_data, mask=True)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask
            
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_idx)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_idx, masked_token_idx)
            # [B, N, P_mask, L, C]

            if train == True:
                reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
                # [B, N, P_mask, L, C] -> [B, P_mask * L * C, N]
            
            return reconstruction_masked_tokens, label_masked_tokens

        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full  # [B, N, P, C]


class EncoderUnmask2Decoder(BaseTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, encoder_layers, num_heads, mlp_ration, En_dropout, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, encoder_layers, num_heads, mlp_ration, En_dropout)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode
        self.posE_method = posE_method

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        self.decoder = TConvTADecoder(embed_dim, embed_dim, in_channel*patch_length, 3, 2, num_tokens)
        self.out_norm = "layer"
        self.decoder_norm = norm_layer(self.out_norm, in_channel * patch_length)
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding , -.1, .1)
        trunc_normal_(self.mask_token, std=.1)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        if self.out_norm in ["batch2d", "batch1d"]:
            hidden_states_full = hidden_states_full.permute(0, 3, 1, 2)  
            hidden_states_full = self.decoder_norm(hidden_states_full)  
            hidden_states_full = hidden_states_full.permute(0, 2, 3, 1)  
        else:
            hidden_states_full = self.decoder_norm(hidden_states_full)  
        return hidden_states_full
        
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
        if self.run_mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx)
            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            return hidden_states_full  # [B, N, P, C]


class EncoderUnmask2TransformerDecoder(BaseTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode
        self.posE_method = posE_method
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, in_channel * patch_length, de_layers, de_mlp, de_heads, de_norm, de_dropout)

        self.initialize_weights()

    def initialize_weights(self):
            if self.posE_method == "param":
                nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
            nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)

        if self.run_mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask
            hidden_states_unmasked = self.e2d_embed(hidden_states_unmasked)

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                  unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]


class EncoderUnmaskConv2TransformerDecoder(TConvEncoderMask):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, t_conv_k, t_conv_d, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, t_conv_k, t_conv_d, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."

        self.run_mode = run_mode
        self.posE_method = posE_method

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)
        
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full

    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
        if self.run_mode == "pre-train":
            hidden_states_full, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            hidden_states_full = self.e2d_embed(hidden_states_full)
            hidden_states_full = self.decoding(hidden_states_full, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]


class EncoderFull2TransformerDecoder(EncoderFull):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."

        self.run_mode = run_mode
        self.posE_method = posE_method

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)

        self.initialize_weights()
    
    def initialize_weights(self):
            if self.posE_method == "param":
                nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
            nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_full: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_full [B, N, P, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_full.size(0), hidden_states_full.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_full.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full[:, :, masked_token_idx] = hidden_states_full[:, :, masked_token_idx] + hidden_states_masked

        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        hidden_states_full = self.out_layer(hidden_states_full)
        return hidden_states_full
    
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
        if self.run_mode == "pre-train":
            hidden_states_full, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            hidden_states_full = self.e2d_embed(hidden_states_full)
            hidden_states_full = self.decoding(hidden_states_full, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]


class EncoderFullConv2TransformerDecoder(TConvEncoderFull):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, t_conv_k, t_conv_d, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, t_conv_k, t_conv_d, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."

        self.run_mode = run_mode
        self.posE_method = posE_method

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)
        
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_full: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_full [B, N, P, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_full.size(0), hidden_states_full.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_full.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full[:, :, masked_token_idx] = hidden_states_full[:, :, masked_token_idx] + hidden_states_masked

        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full
    
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
        if self.run_mode == "pre-train":
            hidden_states_full, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            hidden_states_full = self.e2d_embed(hidden_states_full)
            hidden_states_full = self.decoding(hidden_states_full, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]


class EncoderUnmask2DecoderLinear(EncoderUnmask2TransformerDecoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode)
        self.decoder = LinearDecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_dropout)

    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        return super().forward(history_data, future_data, epoch, batch_seen, train)


# ====================================================================================================================================
# 基于单独显式TAttn的预训练模型，略好于已发现的base。
class STEncoderUnmask2TransformerDecoder(SingleTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode
        self.posE_method = posE_method
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full
    
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)

        if self.run_mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask
            hidden_states_unmasked = self.e2d_embed(hidden_states_unmasked)

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]


# 基于堆叠式TAttn+SAttn的预训练模型
class ST_EncoderUnmask2TransformerDecoder(STAStackEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode
        self.posE_method = posE_method
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)
        
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full
    
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)

        if self.run_mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask
            hidden_states_unmasked = self.e2d_embed(hidden_states_unmasked)

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]

# 基于平行式TAttn+SAttn的预训练模型
class STParallel_EncoderUnmask2TransformerDecoder(STAConcatEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_dropout, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode
        self.posE_method = posE_method
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)
        
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full
    
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)

        if self.run_mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask
            hidden_states_unmasked = self.e2d_embed(hidden_states_unmasked)

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False)
            
            return hidden_states_full  # [B, N, P, C]


# 基于TAttn+SConv的预训练模型
class SConv_EncoderUnmask2TransformerDecoder(SCTAEncoder):
    def __init__(self, patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, 
                 en_sconv_k, en_sconv_d, en_dropout, en_norm,
                 de_layers, de_heads, de_mlp, de_dropout, de_norm, run_mode):
        
        super().__init__(patch_length, in_channel, embed_dim, patch_embed_norm, posE_dropout, posE_method, num_tokens, mask_ration, en_layers, en_heads, en_mlp, en_sconv_k, en_sconv_d, en_dropout, en_norm)
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode
        self.posE_method = posE_method
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.e2d_embed = MLP((embed_dim, embed_dim), act_type="linear")
        self.decoder = BaseTADecoder(embed_dim, embed_dim, embed_dim, de_layers, de_mlp, de_heads, de_norm, de_dropout)
        self.initialize_weights()

    def initialize_weights(self):
        if self.posE_method == "param":
            nn.init.uniform_(self.posE.embedding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: torch.Tensor):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1)

        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), hidden_states_unmasked.size(-1))  # [B, N, P_mask, C]
        hidden_states_masked = self.posE(mask_token, index=masked_token_idx)

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]
        
        hidden_states_full = self.decoder(hidden_states_full)    # [B, N, P, L*C])

        return hidden_states_full
    
    def forward(self,  history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)

        if "graph" in kwargs:
            graph = kwargs["graph"]  # np.ndarray([N, N])
        else:
            raise ValueError("Where is my graph file?")

        if self.run_mode == "pre-train":
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = super().forward(history_data, mask=True, graph=graph)
            # tensor of [B, N, P_unmask, C], list with length P_unmask, list with length P_mask
            hidden_states_unmasked = self.e2d_embed(hidden_states_unmasked)

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                 unmasked_token_idx, masked_token_idx)
            if train == True:
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)
            
            return reconstruct_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = super().forward(history_data, mask=False, graph=graph)
            
            return hidden_states_full  # [B, N, P, C]
        

class PreTrainModel(nn.Module):
    def __init__(self, encoder_model: BaseTAEncoder, encoder_args,
                       decoder_model, decoder_args,
                       e2d_model, e2d_args,
                       run_mode) -> nn.Module:
        super().__init__()
        assert run_mode in ["pre-train", "forecasting"], "Error run mode."
        self.run_mode = run_mode

        self.load_encoder(encoder_model, encoder_args)
        self.load_decoder(decoder_model, decoder_args)
        self.load_e2d(e2d_model, e2d_args)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_args["in_dim"]))
        self.init_mask_token()

    def init_mask_token(self):
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def load_encoder(self, encoder_name: nn.Module, encoder_args: dict):
        self.encoder = encoder_name(**encoder_args)

    def load_decoder(self, decoder_name: nn.Module, decoder_args: dict):
        self.decoder = decoder_name(**decoder_args)

    def load_e2d(self, e2d_model: nn.Module, e2d_args: dict):
        self.e2d = e2d_model(**e2d_args)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        raise NotImplementedError()
    
    def decoding(self, hidden_states_unmasked: torch.Tensor, masked_token_idx: List, unmasked_token_idx: List):
        """
        Decoding process including encoder 2 decoder, add mask tokens, put it into transformer layers, predict result.
        
        param: hidden_states_unmasked [B, N, P_unmask, C]
        param: masked_token_idx, list of length P_masked.
        param: unmasked_token_idx, list of length P_unmasked.

        return: reconstruction_full [B, N, P(= P_unmask + P_mask), L * C]
        """
        B, N , D = hidden_states_unmasked.size(0), hidden_states_unmasked.size(1), hidden_states_unmasked.size(-1)
        mask_token = self.mask_token.expand(B, N, len(masked_token_idx), D)  # [B, N, P_mask, C]
        
        hidden_states_masked = self.encoder.posE(mask_token, index=masked_token_idx)
        
        # hidden_states_full = torch.zeros(size=(B, N, len(masked_token_idx) + len(unmasked_token_idx), D), dtype=hidden_states_unmasked.dtype, device=hidden_states_unmasked.device)
        # hidden_states_full[:, :, masked_token_idx] = hidden_states_masked
        # hidden_states_full[:, :, unmasked_token_idx] = hidden_states_unmasked
        
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # [B, N, P, C]

        hidden_states_full = self.decoder(hidden_states_full)

        return hidden_states_full


class PreTrainEncoderDecoder(PreTrainModel):
    def __init__(self, encoder_model, encoder_args, decoder_model, decoder_args, e2d_model, e2d_args, run_mode) -> nn.Module:
        super().__init__(encoder_model, encoder_args, decoder_model, decoder_args, e2d_model, e2d_args, run_mode)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, epoch: int = None, batch_seen: int = None, train: bool = True, **kwargs):
        history_data = history_data.permute(0, 2, 3, 1)  # [B, L * P, N, C] -> [B, N, C, L * P]
        B, N = history_data.size(0), history_data.size(1)
    
        if self.run_mode == "pre-train":
            if "graph" in kwargs:
                graph = kwargs["graph"]  # np.ndarray([N, N])
            else:
                raise ValueError("Where is my graph file?")
    
            hidden_states_unmasked, unmasked_token_idx, masked_token_idx = self.encoder(history_data, mask=True, graph=graph)
            hidden_states_unmasked = self.e2d(hidden_states_unmasked)

            hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_idx, unmasked_token_idx)

            reconstruct_masked_tokens, label_masked_tokens = self.encoder.get_reconstructed_masked_tokens(hidden_states_full, history_data, 
                                                                                                          unmasked_token_idx, masked_token_idx)
            if train == True:  # for plotting the original masked data in figures.
                reconstruct_masked_tokens = reconstruct_masked_tokens.view(B, N, -1).transpose(1, 2)
                label_masked_tokens = label_masked_tokens.view(B, N, -1).transpose(1, 2)  # [B, N, P, C]
            return reconstruct_masked_tokens, label_masked_tokens
        
        else:
            if "backbone_graph" in kwargs:
                graph = kwargs["backbone_graph"]  # np.ndarray([N, N])
            else:
                raise ValueError("Where is my graph file?")
            hidden_states_full, _, _ = self.encoder(history_data, mask=False, graph=graph)
        
            return hidden_states_full  # [B, N, P, C]






if __name__ == "__main__":
    backbone = BackBone(patch_length=12, in_channel=2, embed_dim=8, num_heads=1, mlp_ratio=4, dropout=0.1,
                        num_token=6, mask_ratio=0.4, encoder_depth=2, decoder_depth=2, mode='pre-train')
    
    stformer = STFormer(patch_length=12, in_channel=2, embed_dim=8, num_heads=1, mlp_ratio=4, dropout=0.1,
                        num_token=6, mask_ratio=0.4, encoder_depth=2, decoder_depth=2, mode='pre-train')
    history_data = torch.randn(32, 12*6, 150, 2)  # [B, L*P, N, C]

    reconstruct, label = stformer(history_data, future_data=None, epoch=0, batch_seen=0, train=True)

    print(reconstruct.size())
    print(label.size())
