# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/25
 
@Author : Shen Fang
"""

import os
import math
import torch

from basic_ts.data.registry import SCALER_REGISTRY
from basic_ts.runners import BaseTimeSeriesForecastingRunner
from basic_ts.utils import load_adj


class MyDataRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        graph = None if cfg.get("GRAPH_PATH", None) is None else load_adj(cfg.get("GRAPH_PATH"), "original")
        self.graph = graph[1] if graph is not None else None

    @staticmethod
    def organize_dataset_args(cfg: dict, dataset_args: dict = None):
        if dataset_args is None:
            dataset_args = dict()
        
        dataset_args["data_file_path"] = "{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        dataset_args["index_file_path"] = "{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        dataset_args = MyDataRunner.organize_external_data_args(cfg, dataset_args)

        return dataset_args
    
    @staticmethod
    def organize_external_data_args(cfg: dict, dataset_args: dict = None):
        if dataset_args is None:
            dataset_args = dict()

        if cfg.get("DATASET_DESCRIBE").get("poi_file") is not None:
            dataset_args["poi_file"] = os.path.join(cfg["TRAIN"]["DATA"]["DIR"], cfg.get("DATASET_DESCRIBE").get("poi_file"))
        
        if cfg.get("DATASET_DESCRIBE").get("wea_file") is not None:
            dataset_args["wea_file"] = os.path.join(cfg["TRAIN"]["DATA"]["DIR"], cfg.get("DATASET_DESCRIBE").get("wea_file"))

        if cfg.get("DATASET_DESCRIBE").get("time_file") is not None:
            dataset_args["time_file"] = os.path.join(cfg["TRAIN"]["DATA"]["DIR"], cfg.get("DATASET_DESCRIBE").get("time_file"))
        
        return dataset_args
    
    def build_train_dataset(self, cfg: dict):
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args = self.organize_dataset_args(cfg, dataset_args)
    
        dataset_args["mode"] = "train"
        dataset_args["time_interval"] = cfg.get("DATASET_DESCRIBE").get("time_interval")
        dataset_args["one_day_range"] = cfg.get("DATASET_DESCRIBE").get("one_day_range")
        dataset_args["divide_days"] = cfg.get("DATASET_DESCRIBE").get("divide_days")

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    def build_val_dataset(self, cfg: dict):
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args = self.organize_dataset_args(cfg, dataset_args)

        dataset_args["mode"] = "valid"
        dataset_args["time_interval"] = cfg.get("DATASET_DESCRIBE").get("time_interval")
        dataset_args["one_day_range"] = cfg.get("DATASET_DESCRIBE").get("one_day_range")
        dataset_args["divide_days"] = cfg.get("DATASET_DESCRIBE").get("divide_days")

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("valid len: {0}".format(len(dataset)))

        return dataset
    
    def build_test_dataset(self, cfg: dict):
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args = self.organize_dataset_args(cfg, dataset_args)

        dataset_args["mode"] = "test"
        dataset_args["time_interval"] = cfg.get("DATASET_DESCRIBE").get("time_interval")
        dataset_args["one_day_range"] = cfg.get("DATASET_DESCRIBE").get("one_day_range")
        dataset_args["divide_days"] = cfg.get("DATASET_DESCRIBE").get("divide_days")

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset
    
    def select_input_features(self, data: torch.Tensor) -> torch.Tensor: 
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]  # [B, T, N, C]
        return data
    
    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, :, self.target_features]  # [B, T, N, C]
        return data
    
    def forward(self,  data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs)  -> tuple:
        if self.graph is not None:
            kwargs["graph"] = self.graph

        future_data, history_data = data

        history_data = self.select_input_features(history_data)

        history_data = self.to_running_device(history_data)      # B, L * P, N, C
        future_data  = self.to_running_device(future_data)       # B, L, N, C
        
        # feedforward
        prediction = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch, train=train)
        # [B, L, N, C]

        B, L, N, _ = future_data.shape

        assert list(prediction.shape)[:3] == [B, L, N], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        
        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)

        return prediction, real_value