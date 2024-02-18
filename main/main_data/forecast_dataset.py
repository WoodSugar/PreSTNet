# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/23
 
@Author : Shen Fang
"""
import os
import torch
from torch.utils.data import Dataset
from basic_ts.utils import load_pkl
from basic_ts.data.dataset import TimeSeriesForecastingDataset
from .my_dataset import MyDataset


class ForecastDataset(MyDataset):
    def __init__(self, data_file_path: str, index_file_path: str, mode: str, seq_len:int, **kwargs):
        super().__init__(data_file_path, index_file_path, mode)

        # length of long term historical data
        self.seq_len = seq_len
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])

    def __getitem__(self, index: int) -> tuple:
        idx = list(self.index[index])    
        
        history_data = self.data[idx[0]:idx[1]]  # 6
        future_data = self.data[idx[1]:idx[2]]   # 6
        if idx[1] - self.seq_len < 0:
            long_history_data = self.mask
        else:
            long_history_data = self.data[idx[1]-self.seq_len: idx[1]] # 288 = 16 * 6 * 3

        return future_data, history_data, long_history_data

    def __len__(self):
        return len(self.index)