import os
import torch
import numpy as np

from easytorch.utils.dist import master_only
from basic_ts.data.registry import SCALER_REGISTRY

from model_utils import init_weights
from .mydata_runner import MyDataRunner


class PreTrainRunner(MyDataRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # self.model.apply(init_weights)
        self.cfg = cfg
        
    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        # return super().forward(data, epoch, iter_num, train, **kwargs)
        future_data, history_data = data
        if self.graph is not None:
            kwargs["graph"] = self.graph
        else:
            kwargs["graph"] = None
    
        history_data = self.select_input_features(history_data)
        future_data = self.select_target_features(future_data)
        # B, T, N, C = future_data.size()
        history_data = self.to_running_device(history_data)  # B, T, N, C
        future_data = self.to_running_device(future_data)  # B, T, N, C

        reconstruction_masked_tokens, label_masked_tokens = self.model(history_data=history_data, future_data=future_data, epoch=epoch, batch_seen=iter_num, train=train, **kwargs)
        return reconstruction_masked_tokens, label_masked_tokens

    @master_only
    @torch.no_grad()
    def test(self):
        for data_idx, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=True)

            # re-scale data for testing the reconstruction accuracy
            prediction_rescale = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
            real_value_rescale = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
           
            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(prediction_rescale, real_value_rescale, null_val=self.null_val)
                self.update_epoch_meter("test_" + metric_name, metric_item.item())

    def on_training_end(self):
        super().on_training_end()
        if self.test_data_loader is not None:
            for data_idx, data in enumerate(self.test_data_loader):
                forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
                
                # re-scale data for testing the reconstruction accuracy
                prediction_rescale = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
                real_value_rescale = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
                # [B, N, P_mask, L, C]
                B, N, P_mask, L, C = prediction_rescale.size()
                if data_idx % 5 == 0:
                    self.plt_long_history(SCALER_REGISTRY.get(self.scaler["func"])(data[1], **self.scaler["args"]), prediction_rescale, real_value_rescale, data_idx)
                    self.save_mask_data(SCALER_REGISTRY.get(self.scaler["func"])(data[1], **self.scaler["args"]), prediction_rescale, real_value_rescale, data_idx)

    def save_mask_data(self, long_history_data: torch.Tensor, mask_pred: torch.Tensor, mask_target: torch.Tensor, data_idx: int):
        """
        :param history_data: [B, T=(P*L), N, C]
        :param pred: [B, N, P_mask, L, C]
        :param target: [B, N, P_mask, L, C]
        """
        if self.cfg.GPU_NUM <= 1:
            L = self.model.encoder.patch_length
            P = self.model.encoder.num_tokens
            token_ids = self.model.encoder.mask.masked_tokens
        else:
            L = self.model.module.encoder.patch_length
            P = self.model.module.encoder.num_tokens
            token_ids = self.model.module.encoder.mask.masked_tokens

        # real_value_full [B, L*P, N, C] -> [B, P, N, C, L] -> [B, N, P, L, C]
        long_history_data = long_history_data.unfold(dimension=1, size=L, step=L).permute(0, 2, 1, 4, 3).detach().cpu().numpy()
        
        mask_pred = mask_pred.detach().cpu().numpy()      # [B, N, P_mask, L, C]
        mask_target = mask_target.detach().cpu().numpy()  # [B, N, P_mask, L, C]

        mask_tokens = np.concatenate([[np.array(range(each_id * L, (each_id + 1) * L))] for each_id in token_ids], axis=0)
        
        output_path = os.path.join(self.ckpt_save_dir, self.cfg.MODEL.NAME + f"{data_idx}" + ".npz")

        data_dict= {"long_history_data": long_history_data,
                    "mask_pred":         mask_pred,
                    "mask_target":       mask_target,
                    "tokens:":           mask_tokens}

        np.savez(output_path, **data_dict)

        

    def plt_long_history(self, long_history_data: torch.Tensor, mask_pred: torch.Tensor, mask_target: torch.Tensor, data_idx: int):
        """
        :param history_data: [B, T=(P*L), N, C]
        :param pred: [B, N, P_mask, L, C]
        :param target: [B, N, P_mask, L, C]
        """
        if self.cfg.GPU_NUM <= 1:
            L = self.model.encoder.patch_length
            P = self.model.encoder.num_tokens
            token_ids = self.model.encoder.mask.masked_tokens
        else:
            L = self.model.module.encoder.patch_length
            P = self.model.module.encoder.num_tokens
            token_ids = self.model.module.encoder.mask.masked_tokens

        # real_value_full [B, L*P, N, C] -> [B, P, N, C, L] -> [B, N, P, L, C]
        long_history_data = long_history_data.unfold(dimension=1, size=L, step=L).permute(0, 2, 1, 4, 3)
        
        batch_idx = 2
        node_idx = 10
        patch_idx = 0
        channel_idx = 0
        
        real_value = long_history_data[batch_idx, node_idx, :, :, channel_idx] # [P, L]
        real_value = torch.cat([real_value[i, :] for i in range(P)], dim=0).detach().cpu().numpy()  # [P * L]
        
        pred_rescale = mask_pred[batch_idx, node_idx, :, :, channel_idx].detach().cpu().numpy()  # [P_mask, L]
        # pred_rescale = torch.cat([pred_rescale[i, :] for i in range(pred_rescale.size(0))]).detach().cpu().numpy()  # [P_mask * L]
        
        # real_value_rescale = target[batch_idx, node_idx, patch_idx, :, channel_idx].detach().cpu().numpy()
        
        real_x_axis = np.array(list(range(real_value.shape[0])))
        pred_x_axis = np.concatenate([np.array(range(token_id * L, (token_id + 1) * L)) for token_id in token_ids], axis=0)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.grid(True, linestyle="-.", linewidth=0.5)
        
        plt.plot(real_x_axis, real_value, ls="-", label="real value", color="g")
        for i, token_id in enumerate(token_ids):
            plt.plot(np.array(range(token_id * L, (token_id + 1) * L)), pred_rescale[i, :], ls="-", label="reconstruction", color="r")
            # plt.plot(pred_x_axis, pred_rescale, ls="-", label="reconstruction", color="r")

        # plt.legend(["real value", "reconstruction"], loc="upper right")
        plt.xlabel("time axis")
        plt.ylabel("traffic flow")
        plt.title("prediction vs real value")
        plt.savefig(self.ckpt_save_dir + "/long_history_{:03d}.png".format(data_idx))
    
    def plt_recover_data(self, history_data: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, data_idx: int):
        L = self.model.encoder.patch_length
        P = self.model.encoder.num_tokens
        history_data = history_data.unfold(dimension=1, size=L, step=L).permute(0, 2, 1, 4, 3)  # [B, N, P, L, C]

        batch_idx = 10
        node_idx = 10
        patch_idx = 0
        channel_idx = 1

        real_value_rescale = target[batch_idx, node_idx, patch_idx, :, channel_idx].detach().cpu().numpy()
        pred_rescale = pred[batch_idx, node_idx, patch_idx, :, channel_idx].detach().cpu().numpy()  # [P_mask, L]

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.grid(True, linestyle="-.", linewidth=0.5)

        plt.plot(real_value_rescale, ls="-", label="real value", color="g")
        plt.plot(pred_rescale, ls="-", label="reconstruction", color="r")
        
        plt.xlabel("time axis")
        plt.ylabel("traffic flow")
        plt.title("prediction vs real value")
        plt.savefig(self.ckpt_save_dir + "/short_patch_{:03d}.png".format(data_idx))