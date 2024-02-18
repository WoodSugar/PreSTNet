# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/15

@Author : Shen Fang
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import argparse
import os
import json

import warnings
from typing import Union, List, Tuple


class CreateOption:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch implementation of Traffic Prediction")

        # dataset args
        parser.add_argument("--data_folder", type=str, help="folder of the dataset")
        parser.add_argument("--data_path", type=dict, help="data path of the dataset")
        parser.add_argument("--divide_days", type=list, help="divide the dataset into three parts")
        parser.add_argument("--one_day_range", type=list, help="hour range of one day")
        parser.add_argument("--time_interval", type=int, help="minute intervals of two successive data.")
        parser.add_argument("--merge_num", type=int, help="merging records of two successive data.")
        parser.add_argument("--batch_size", type=int, default=32, help=" batch size")
        parser.add_argument("--num_nodes", type=int, help="number of nodes")
        parser.add_argument("--input_dim", type=int, help="the number of input channel dimension")
        parser.add_argument("--src_len", type=int, help="length of the history data will be used")
        parser.add_argument("--trg_len", type=int, help="length of the future data will be predicted")
        parser.add_argument("--predict_idx", type=int, default=[0,1,2,3,4,5], help="prediction index")

        # train args
        parser.add_argument("--save_model", type=bool, default=True, help="whether to save the model result")
        parser.add_argument("--save_mode", type=str, default="best", help="save the best model or each epoch")
        parser.add_argument("--epoch", type=int, help="train epoch")
        parser.add_argument("--log", type=str, help="basic name of the model")

        # result args
        parser.add_argument("--result_folder", type=str, help="result folder?")

        args = parser.parse_args()
        args.data_path = dict()

        self.args = args

    def add_data_folder(self, data_folder):
        self.args.data_folder = data_folder

    def add_data_path(self, data_key, data_path):
        self.args.data_path[data_key] = data_path

    def add_result_folder(self, folder):
        self.args.result_folder = folder

    def add_log_name(self, log):
        self.args.log = log

    def initialize(self):
        if not os.path.exists(self.args.result_folder):
            os.mkdir(self.args.result_folder)

        self.args.log = os.path.join(self.args.result_folder, self.args.log)

        return self.args


def create_option(data_info_path: str, result_folder: str, log_name: str, model_hyper: dict):

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    data_info = json.load(open(data_info_path, "r"))

    option_class = CreateOption()

    option_class.add_data_folder(data_info["folder"])

    option_class.add_data_path("flow", data_info["flow_file"])
    option_class.add_data_path("graph", data_info["graph_file"])

    if "poi_file" in data_info:
        option_class.add_data_path("poi", data_info["poi_file"])

    if "wea_file" in data_info:
        option_class.add_data_path("wea", data_info["wea_file"])

    if "time_file" in data_info:
        option_class.add_data_path("time", data_info["time_file"])

    option_class.add_result_folder(result_folder)
    option_class.add_log_name(log_name)

    option = option_class.initialize()

    option.num_nodes = data_info["num_nodes"]
    option.divide_days = data_info["divide_days"]
    option.one_day_range = data_info["one_day_range"]
    option.time_interval = data_info["time_interval"]
    option.merge_num = data_info["merge_num"]

    option.input_dim = data_info["input_dim"]
    option.src_len = data_info["src_len"]
    option.trg_len = data_info["trg_len"]

    option.epoch = model_hyper["epoch"]  # 1500

    option.batch_size = model_hyper["batch_size"]  # 128, 16, 8, 4

    option.predict_idx = model_hyper["predict_idx"]  # [0, 1, 2, 3, 4, 5]

    return option


class Evaluate:
    def __init__(self, axis):
        self.axis = axis

    def mae_(self, predict, target):
        """
        :param predict: [B, N, TRG_len, C]  B = Time Axis
        :param target:  [B, N, TRG_len, C]
        :return:        [B, N, TRG_len, C]
        """
        return np.mean(np.abs(target - predict), axis=self.axis)

    def mape_(self, predict, target, ep=5):
        return np.mean(np.abs(target - predict) / (target + ep), axis=self.axis)

    def rmse_(self, predict, target):
        return np.sqrt(np.mean(np.power(target - predict, 2), axis=self.axis))

    def smape_(self, predict, target, ep=1):
        pass
        smape = np.mean(np.abs(predict - target) / (np.abs(predict) + np.abs(target) + 2*ep) / 2, axis=self.axis)


    def total(self, predict, target, ep=5):
        return self.mae_(predict, target), self.mape_(predict, target, ep), self.rmse_(predict, target)
    

def choose_data_one_day(data: np.array, time_interval: int, one_day_range: Union[List[int], Tuple[int]], total_days: int) -> np.array:
    """
    Choose data from one day.
    Args:
        :param data: 3D array, shape=(T, N, C)
        :param time_interval: int, time interval of data
        :param one_day_range: list or tuple, range hour of one day [s, e]
        :param total_days: int, total days of data
    Returns:
        :return: 3D array, shape=(T1, N, C)
    """
    records_each_hour = int(60 / time_interval)
    data_list =  [data[(day * 24 + one_day_range[0]) * records_each_hour: 
                       (day * 24 + one_day_range[1]) * records_each_hour]  for day in range(total_days)]
    return np.concatenate(data_list, axis=0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_parameters(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def fast_power(matrix_x, pow_n):
    """
    calculate the fast power of matrix.

    :param matrix_x: the matrix, [N, N]
    :param pow_n: the power, int
    :return: result of matrix_x^pow_n
    """

    result = torch.eye(matrix_x.size(0), device=matrix_x.device)

    if pow_n < 1:
        return result
    elif pow_n == 1:
        return matrix_x

    while pow_n:
        if pow_n & 1:
            result = torch.mm(result, matrix_x)
        matrix_x = torch.mm(matrix_x, matrix_x)

        pow_n >>= 1

    return result


def plot_curve(result_file, x_range, y_range):

    train_file = result_file + "_train.csv"
    valid_file = result_file + "_valid.csv"

    train_log = pd.read_csv(train_file, header=None, usecols=[0, 1]).values
    valid_log = pd.read_csv(valid_file, header=None, usecols=[0, 1]).values

    plt.figure()
    plt.grid(True, linestyle="-.", linewidth=0.5)

    plt.plot(train_log[:, 0], train_log[:, 1], ls="-", marker=" ", color="r")
    plt.plot(valid_log[:, 0], valid_log[:, 1], ls="-", marker=" ", color="g")

    plt.legend(["train loss", "valid loss"], loc="upper right")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Results")

    plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])

    plt.savefig(result_file + ".png")


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def sum_byindex(a: np.ndarray, index: List, axis: int):
    if axis != 0 :
        a = a.transpose((axis,0))
    index_arr = np.concatenate(index)
    lens = np.array([len(i) for i in index])
    cut_idx = np.concatenate(([0], lens[:-1].cumsum() ))
    a = np.add.reduceat(a[index_arr], cut_idx)
    
    if axis != 0 :
        a = a.transpose((axis,0))
    
    return a
