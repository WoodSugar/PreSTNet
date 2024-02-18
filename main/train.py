# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/26

@Author : Shen Fang
"""
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))

from argparse import ArgumentParser
import torch
from basic_ts import launch_training

torch.set_num_threads(4)  # aviod high cpu avg usage


def create_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default='model_setting/pre_train/PeMS03ProposeMAEE100.py', help="training config file path.")
    # parser.add_argument("-c", "--cfg", default='model_setting/EncoderUnmaks2TransformerDecoder_MyBus.py', help="training config file path.")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus id.")

    return parser.parse_args()


if __name__ == "__main__":
    args = create_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    launch_training(args.cfg, args.gpus)
