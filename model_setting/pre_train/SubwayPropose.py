# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/11
 
@Author : Shen Fang
"""

import os
import json

from easydict import EasyDict

from basic_ts.losses import masked_mae
from main.main_data.pretrain_dataset import PreTrainDataset
from main.main_runner.pretrain_runner import PreTrainRunner

from main.main_arch.backbone import PreTrainEncoderDecoder

from main.main_arch.module import SCTAEncoder
from main.main_arch.module import BaseTADecoder
from model_utils import MLP 


CFG = EasyDict()
# ================= general ================= #
CFG.DESCRIPTION = "PreTrain(MySubway) configuration"
CFG.RUNNER = PreTrainRunner
CFG.DATASET_CLS = PreTrainDataset
CFG.DATASET_NAME = "MySubway"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_DESCRIBE = EasyDict(json.load(open(os.path.join("dataset_describe", CFG.DATASET_NAME), "r")))

CFG.DATASET_INPUT_LEN = CFG.DATASET_DESCRIBE.get("pretrain_src_len")
CFG.DATASET_OUTPUT_LEN = CFG.DATASET_DESCRIBE.get("pretrain_trg_len")
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "SubwayPreTrainPropose"
CFG.MODEL.ARCH = PreTrainEncoderDecoder
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0, 1]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.FORWARD_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)

CFG.GRAPH_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), "adj_mx.pkl")

CFG.MODEL.PARAM = {
    "encoder_model": SCTAEncoder, 
    "encoder_args": {
        "patch_length": 6, 
        "in_channel": CFG.MODEL.INPUT_DIM, 
        "embed_dim": 64, 
        "patch_embed_norm": "layer", 
        "posE_dropout": 0, 
        "posE_method": "param",
        "mask_ration": 0.75,
        "n_layers": 4, 
        "num_heads": 4, 
        "mlp_ration": 2, 
        "en_sconv_k": 3,
        "en_sconv_d": [1,2,4],
        "en_dropout": 0, 
        "en_norm": "layer"
    }, 

    
    "decoder_model": BaseTADecoder, 
    "decoder_args": {
        "in_dim": 64, 
        "hid_dim": 64, 
        "n_layers": 1, 
        "mlp_ration": 2, 
        "num_heads": 4, 
        "norm_method": "layer", 
        "dropout": 0
    }, 
    
    "e2d_model": MLP, 
    "e2d_args": {
        "channels": (64, 64),
        "act_type": "linear"
    },
    "run_mode": "pre-train"
}

CFG.MODEL.PARAM["encoder_args"]["num_tokens"] =  CFG.DATASET_DESCRIBE.get("pretrain_src_len") // CFG.MODEL.PARAM["encoder_args"]["patch_length"]
CFG.MODEL.PARAM["decoder_args"]["out_dim"] =  CFG.MODEL.PARAM["encoder_args"]["patch_length"] * CFG.MODEL.INPUT_DIM


# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1.5e-3,
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
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 500
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "../data/ST_Pre_Result",
    "_".join([CFG.MODEL.NAME, CFG.DATASET_NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = CFG.DATASET_DESCRIBE.get("folder")
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False

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
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = False