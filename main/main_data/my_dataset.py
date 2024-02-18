import os
import numpy as np

import torch
from torch.utils.data import Dataset
from data_loader import LoadData

from basic_ts.data.dataset import TimeSeriesForecastingDataset
from utils import choose_data_one_day


class MyDataset(TimeSeriesForecastingDataset):
    def __init__(self, data_file_path: str, index_file_path: str, mode: str, **kwargs) -> None:
        super().__init__(data_file_path, index_file_path, mode)
        T, N, C = self.data.size()
        
        if "poi_file" in kwargs:
            poi_data = np.load(kwargs["poi_file"])  # [N, D]

            poi_max, poi_min = LoadData.normalize_base(poi_data, 0)
            poi_data = LoadData.normalize_data(poi_max, poi_min, poi_data)
            
            poi_data = torch.from_numpy(poi_data).float().unsqueeze(0).expand(T, -1, -1)  # [T, N, D]
            self.data = torch.concat([self.data, poi_data], dim=-1)  # [T, N, D1 + D2]

        if "wea_file" in kwargs:
            wea_data = np.load(kwargs["wea_file"])  # [SeqT, D]
            wea_data = choose_data_one_day(wea_data, kwargs["time_interval"], kwargs["one_day_range"], sum(kwargs["divide_days"]))
            wea_max, wea_min = LoadData.normalize_base(wea_data, 0)
            wea_data = LoadData.normalize_data(wea_max, wea_min, wea_data)
            wea_data = torch.from_numpy(wea_data).float().unsqueeze(1).expand(-1, N, -1)  # [SeqT, N, D]
            self.data = torch.concat([self.data, wea_data], dim=-1)  # [T, N, D1 + D2]

        if "time_file" in kwargs:
            time_data = np.load(kwargs["time_file"])  # [SeqT, D]
            time_data = choose_data_one_day(time_data, kwargs["time_interval"], kwargs["one_day_range"], sum(kwargs["divide_days"]))
            time_max, time_min = LoadData.normalize_base(time_data, 0)
            time_data = LoadData.normalize_data(time_max, time_min, time_data)
            time_data = torch.from_numpy(time_data).float().unsqueeze(1).expand(-1, N, -1)  # [SeqT, N, D]
            self.data = torch.concat([self.data, time_data], dim=-1)  # [T, N, D1 + D2]

    def __getitem__(self, index):        
        return super().__getitem__(index)  # [L, N, C]

    def __len__(self):
        return super().__len__()