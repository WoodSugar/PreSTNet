import torch 
import torch.nn as nn

from .backbone import EncoderUnmask2TransformerDecoder
from .head import MLPLastLongHeader
from model_utils import MLP 



class AuxCompute(nn.Module):
    def __init__(self, source_in_dim:int, target_in_dim: int, target_seq:int=None):
        super().__init__()
        if source_in_dim == target_in_dim:
            self.merge = nn.Identity()
        else:
            self.merge = MLP((target_in_dim, source_in_dim), act_type=None)

    def forward(self, long_term_history: torch.Tensor, backbone: nn.Module, **kwargs):
        """
        :param long_term_history, [B, L * P, N, C]
        :return  # [B, N, P, C]
        """
        device = long_term_history.device
        B, _, N, _ = long_term_history.shape  # [B, L*P, N ,C]
        
        hidden_states = backbone(self.merge(long_term_history), **kwargs)

        return hidden_states


class ForecastModel(nn.Module):
    def __init__(self, dataset_name, backbone_path, 
                 backbone_model, backbone_args, 
                 predictor_model, predictor_args, 
                 aux_compute_args):

        super().__init__()
        self.dataset_name = dataset_name
        self.backbone_path = backbone_path

        # initalize backbone and load pre-trained params
        self.load_back_bone(backbone_model, backbone_args)

        # initalize prediction header
        self.load_predictor_head(predictor_model, predictor_args)

        self.aux_compute = AuxCompute(**aux_compute_args)

    def load_back_bone(self, backbone_name: nn.Module, backbone_args: dict):
        self.backbone = backbone_name(**backbone_args)

        checkpoint_dict = torch.load(self.backbone_path)
        self.backbone.load_state_dict(checkpoint_dict["model_state_dict"])

        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_predictor_head(self, predictor_name: nn.Module, predictor_args: dict):
        self.pred_header = predictor_name(**predictor_args)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs):
        raise NotImplementedError()


class HeaderWithBackbone(ForecastModel):
    def __init__(self, dataset_name, backbone_path, backbone_model, backbone_args, predictor_model, predictor_args, aux_compute_args):
        super().__init__(dataset_name, backbone_path, backbone_model, backbone_args, predictor_model, predictor_args, aux_compute_args)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train:bool, **kwargs):
        short_term_history = history_data     # [B, L, N, C]
        long_term_history = long_history_data # [B, L*P, N, C]

        B, L, N, C = short_term_history.size()

        hidden_states = self.aux_compute(long_term_history, self.backbone, **kwargs)  # [B, N, P, C]
        prediction = self.pred_header(short_term_history, hidden_states, **kwargs)  # [B, L, N, C]

        return prediction
