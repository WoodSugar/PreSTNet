# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/12

@Author : Shen Fang
"""
# Summary: 检测当前Pytorch和设备是否支持CUDA和cudnn

import torch

if __name__ == '__main__':
    print("Support CUDA ?: ", torch.cuda.is_available())
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print(xx)

    y = torch.randn(2, 3)
    yy = y.cuda()
    print(yy)

    zz = xx + yy
    print(zz)

    # CUDNN TEST
    from torch.backends import cudnn

    print("Support cudnn ?: ", cudnn.is_acceptable(xx))
