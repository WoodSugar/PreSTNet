# -*- coding: utf-8 -*-
""" 
@Time   : 2023/11/17
 
@Author : Shen Fang
"""
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F


class MetaLearner(nn.Module):
    def __init__(self, meta_in_c, meta_W_in, meta_W_out):
        super(MetaLearner, self).__init__()
        self.meta_W = nn.Linear(meta_in_c, meta_W_in * meta_W_out, bias=True)
        self.meta_b = nn.Linear(meta_in_c, meta_W_out, bias=True)
        self.W_in = meta_W_in
        self.W_out = meta_W_out

    def forward(self, meta_knowledge: torch.Tensor):
        """
        :param node_mk: [*X*, meta_in_c], meta knowledge, *X* means the previous dimension.
        :return:
            meta_W: [*X*, meta_W_in, meta_W_out].
            meta_b: [*X*, meta_W_out].
        """
        meta_W = self.meta_W(meta_knowledge)  # meta_know: [B, N, D] -> W [B, N, W_in, W_out]
        meta_b = self.meta_b(meta_knowledge)  # meta_know: [B, N, D] -> b [B, N, W_out]

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



class SMeta(nn.Module):
    def __init__(self, ref_dim:int, in_dim:int, hid_dim:int, out_dim:int):
        super(SMeta, self).__init__()

        self.mk_learner = MetaLearner(ref_dim * 2, hid_dim * 2, hid_dim)
        self.in_ = nn.Linear(in_dim, hid_dim)
        self.out_ = nn.Linear(hid_dim, out_dim)

    def forward(self, long_data_states:torch.Tensor, short_data_in:torch.Tensor, graph: tuple) -> torch.Tensor:
        """
        :param long_data_hid: [B, N, hid_dim].
        :param short_data_in: [B, L1, N, in_dim].
        :return: [B, L1, N, out_dim].
        """
        device = short_data_in.device
        B, L1, N, in_dim = short_data_in.size()
        short_data_in = short_data_in.permute(2, 0, 1, 3)  # [N, B, L1, in_dim].
        long_data_states = long_data_states.transpose(0, 1)  # [N, B, hid_dim].

        dgl_graph = dgl.graph(graph, device=device)
        
        dgl_graph.ndata["mk"] = long_data_states
        dgl_graph.ndata["state"] = self.in_(short_data_in)

        dgl_graph.update_all(self.message_edge, self.message_reduce)
        state = dgl_graph.ndata.pop("new_state")

        prediction = self.out_(state)  # [N, B, L1, D]
        return prediction.permute(1, 2, 0, 3)

    def message_edge(self, edge):
        mk = torch.cat((edge.src["mk"], edge.dst["mk"]), dim=-1)
        state = torch.cat((edge.src["state"], edge.dst["state"]), dim=-1)

        meta_W, meta_b = self.mk_learner(mk)

        attention = torch.matmul(state, meta_W) + meta_b.unsqueeze(-2)

        return {"attention" : attention, "state": edge.src["state"]}

    def message_reduce(self, node):
        state = node.mailbox["state"]
        attention = node.mailbox["attention"]

        attention = attention.softmax(dim=1)
        new_state = torch.sum(attention * state, dim=1)

        return {"new_state": new_state}


class TMeta(nn.Module):
    def __init__(self, ref_dim: int, in_dim: int, out_dim: int):
        super(TMeta, self).__init__()
        self.mk_learner = MetaLearner(ref_dim, in_dim, out_dim * 2)
        

    def forward(self, long_states, short_data) -> torch.Tensor:
        """
        :param long_data_hid: [B, N, hid_dim].
        :param short_data_in: [B, L1, N, in_dim].
        :return: [B, L1, N, W_out].
        """

        meta_W, meta_b = self.mk_learner(long_states)  # [B, N, W_in, W_Out], [B, N, W_Out]
        short_data = torch.matmul(short_data.permute(2, 0, 1, 3), meta_W.transpose(0, 1))   # [N, B, L1, in_dim] * [N, B, W_in, W_out] = [N, B, L1, W_out]
        short_data = short_data + meta_b.unsqueeze(2).permute(1, 0, 2, 3)
        return F.glu(short_data.permute(1, 2, 0, 3), dim=-1) # [B, L1, N, W_out]


if __name__ == "__main__":
    gcn = SMeta(15, 2, 64, 12)

    long_states = torch.randn(32, 4, 15)  # [B, N, ref_dim]
    short_data = torch.randn(32, 6, 4, 2)  # [B, L, N, C]

    src_id = [0, 0, 1, 2, 3]
    dst_id = [1, 2, 3, 1, 0]


    y = gcn(long_states, short_data, (src_id, dst_id))
    print(y.size())  # [N, B, L1, D]

 
