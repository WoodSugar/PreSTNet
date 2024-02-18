import torch
import numpy as np
import os

from basic_ts.runners import BaseTimeSeriesForecastingRunner
from basic_ts.metrics import masked_mae, masked_rmse, masked_wape, masked_mape
from .pretrain_runner import PreTrainRunner
from model_utils import init_weights

from easytorch.utils.dist import master_only

from .mydata_runner import MyDataRunner
from main.main_loss.my_metrics import masked_smape
from basic_ts.utils import load_adj


class ForecastRunner(MyDataRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "WAPE":  masked_wape, "MAPE": masked_mape})
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "SMAPE": masked_smape}
        # self.model.apply(init_weights)
        self.cfg = cfg

        backbone_graph = None if cfg.get("BackBone_GRAPH", None) is None else load_adj(cfg.get("BackBone_GRAPH"), "original")
        self.backbone_graph = backbone_graph[1] if backbone_graph is not None else None


    def forward(self,  data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs)  -> tuple:
        if self.graph is not None:
            kwargs["graph"] = self.graph
        
        kwargs["backbone_graph"] = self.backbone_graph
        
        
        future_data, history_data, long_history_data = data

        history_data = self.select_input_features(history_data)
        long_history_data = self.select_input_features(long_history_data)

        # history_data        = self.to_running_device(history_data)      # B, L, N, C
        # long_history_data   = self.to_running_device(long_history_data)       # B, L * P, N, C
        # future_data         = self.to_running_device(future_data)       # B, L, N, C
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        history_data        = history_data.to(device)      # B, L, N, C
        long_history_data   = long_history_data.to(device)       # B, L * P, N, C
        future_data         = future_data.to(device)       # B, L, N, C
        
        # feedforward
        prediction = self.model(history_data=history_data, long_history_data=long_history_data, future_data=future_data, batch_seen=iter_num, epoch=epoch, train=train, **kwargs)
        # [B, L, N, C]

        B, L, N, _ = future_data.shape

        assert list(prediction.shape)[:3] == [B, L, N], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        
        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)

        return prediction, real_value
    
    @master_only
    @torch.no_grad()
    def test(self):
        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = list(self.forward(data, epoch=None, iter_num=None, train=False))
            if self.if_evaluate_on_gpu:
                forward_return[0], forward_return[1] = forward_return[0].detach().cpu(), forward_return[1].detach().cpu()
            prediction.append(forward_return[0])        # preds = forward_return[0]
            real_value.append(forward_return[1])        # testy = forward_return[1]
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = self.rescale_data(prediction) if self.if_rescale else prediction
        real_value = self.rescale_data(real_value) if self.if_rescale else real_value
        # evaluate
        self.evaluate(prediction, real_value)  
        
        print(prediction.size(), real_value.size())  # [samples, trg_len, num_nodes, output_dim]
        prediction = prediction.detach().cpu().numpy()
        real_value = real_value.detach().cpu().numpy()


        output_path = os.path.join(self.ckpt_save_dir, self.cfg.MODEL.NAME + ".npz")
        data_dict = {"prediction": prediction, "target": real_value}
        np.savez(output_path, **data_dict)


