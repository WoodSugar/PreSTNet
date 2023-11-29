# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/11
 
@Author : Shen Fang
"""

import os
import json

from easydict import EasyDict

from basic_ts.losses import masked_mae

from main.main_data.forecast_dataset import ForecastDataset
from main.main_runner.forecast_runner import ForecastRunner
from main.main_arch.predictor import HeaderWithBackbone

from main.main_arch.head import TMetaConvSConvHeader
from main.main_arch.backbone import PreTrainEncoderDecoder

CFG = EasyDict()
# ================= general ================= #
CFG.DESCRIPTION = "FineTune(MySubway) configuration"
CFG.RUNNER = ForecastRunner
CFG.DATASET_CLS = ForecastDataset
CFG.DATASET_NAME = "MySubway"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_DESCRIBE = EasyDict(json.load(open(os.path.join("dataset_describe", CFG.DATASET_NAME), "r")))

CFG.DATASET_INPUT_LEN = CFG.DATASET_DESCRIBE.get("src_len")
CFG.DATASET_OUTPUT_LEN = CFG.DATASET_DESCRIBE.get("trg_len")
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

CFG.DATASET_ARGS = {
    "seq_len": CFG.DATASET_DESCRIBE.get("pretrain_src_len")
}

CFG.GRAPH_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), "adj_mx.pkl")
CFG.BackBone_GRAPH = CFG.GRAPH_PATH

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "SubwayEnPropose_DeTMetaConvSConv"
CFG.MODEL.ARCH = HeaderWithBackbone
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0, 1]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.FORWARD_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)


from model_setting.pre_train.SubwayPropose import CFG as BackBone

CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "backbone_path": "backbone_pt/Subway_PreTrain.pt",
    
    "backbone_model": PreTrainEncoderDecoder,
    "backbone_args": BackBone.MODEL.PARAM,
        
    "predictor_model": TMetaConvSConvHeader, 
    "predictor_args": {
        "in_dim": CFG.MODEL.INPUT_DIM,
        "hid_dim": 64, 
        "out_dim": CFG.MODEL.INPUT_DIM, 
        "src_len": CFG.DATASET_INPUT_LEN, 
        "trg_len": CFG.DATASET_OUTPUT_LEN, 
        "t_conv_k": 3,
        "t_conv_d": [1, 2, 4],
        "s_conv_k": 3,
        "s_conv_d": [1, 2, 4],
    },

    "aux_compute_args": {
        "source_in_dim": 2,
        "target_in_dim": 2,
        "target_seq":CFG.DATASET_DESCRIBE.get("pretrain_src_len")
    }
}

CFG.MODEL.PARAM["backbone_args"]["run_mode"] = "forecasting"
CFG.MODEL.PARAM["backbone_args"]["encoder_args"]["num_tokens"] =  CFG.DATASET_DESCRIBE.get("pretrain_src_len") // BackBone.MODEL.PARAM["encoder_args"]["patch_length"]
CFG.MODEL.PARAM["predictor_args"]["num_tokens"] =  CFG.DATASET_DESCRIBE.get("pretrain_src_len") // BackBone.MODEL.PARAM["encoder_args"]["patch_length"]


CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 2e-3,
    "weight_decay": 0,
    "eps": 1.0e-8
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}

CFG.TRAIN.NUM_EPOCHS = 500
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "../data/Finetune_Result",
    "_".join([CFG.MODEL.NAME, CFG.DATASET_NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.NULL_VAL = 0.0
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = CFG.DATASET_DESCRIBE.get("folder")
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True

CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 30
CFG.TRAIN.CL.CL_EPOCHS = 3
CFG.TRAIN.CL.PREDICTION_LENGTH = CFG.DATASET_OUTPUT_LEN


# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1 
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 8
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = True

# ================= eval ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = range(1, CFG.DATASET_OUTPUT_LEN+1)