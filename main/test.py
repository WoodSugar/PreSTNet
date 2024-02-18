# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/26

@Author : Shen Fang
"""
import os 
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))
from easytorch import launch_runner, Runner
from argparse import ArgumentParser


def create_args():
    parser =ArgumentParser(description="EasyTorch for time series forecasting test!")
    parser.add_argument("-c", "--cfg", default='model_setting/compare_method/LR_MySubway.py ', help="training config file path.")
    parser.add_argument("-ck", "--ckpt", default='../data/Forecast_Result/LR_MySubway_500/7d08d6c4b52e2d9efd75fea2ea33b038/LR_best_val_MAE.pt', help="the checkpoint file path. if None, load default ckpt in ckpt save dir.")
    parser.add_argument("-g", "--gpus", default="0", help="gpu ids.")

    return parser.parse_args()


def inference(cfg: dict, runner: Runner, ckpt_path: str):
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    
    # runner.model.eval()
    # runner.setup_graph(cfg=cfg, train=False)
    
    runner.load_model(ckpt_path=ckpt_path)
    
    runner.test_process(cfg)


if __name__ == "__main__":
    args = create_args()

    launch_runner(args.cfg, inference, (args.ckpt,), devices=args.gpus)