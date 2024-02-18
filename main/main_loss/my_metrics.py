import torch
import numpy as np
from train_lib import compute_performance
from utils import Evaluate


def compute_performance(prediction, target, dim):
    eval = Evaluate(dim)
    mae, mape, rmse = eval.total(target, prediction, ep=5)

    return mae, mape, rmse

# def compute_performance(prediction, target, dim):
#     eval = Evaluate(dim)
#     mae, mape, rmse = eval.total(target, prediction, ep=5)

#     return mae, mape, rmse


def my_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values, [B, P_mask * L * C, N]
        labels (torch.Tensor): labels           [B, P_mask * L * C, N]
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    preds = preds.to(torch.device("cpu")).detach().numpy()
    labels = labels.to(torch.device("cpu")).detach().numpy()
    
    eval = Evaluate(dim=(0, 1, 2))
    mae = eval.mae_(preds, labels)
    mae = torch.tensor(mae)

    return mae


def my_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    preds = preds.to(torch.device("cpu")).detach().numpy()
    labels = labels.to(torch.device("cpu")).detach().numpy()
    
    eval = Evaluate(dim=(0, 1, 2))
    mape = eval.mape_(preds, labels)
    mape = torch.tensor(mape)

    return mape


def my_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    preds = preds.to(torch.device("cpu")).detach().numpy()
    labels = labels.to(torch.device("cpu")).detach().numpy()
    
    eval = Evaluate(dim=(0, 1, 2))
    rmse = eval.rmse_(preds, labels)
    rmse = torch.tensor(rmse)

    return rmse


def my_smape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:


    preds = preds.to(torch.device("cpu")).detach().numpy()
    labels = labels.to(torch.device("cpu")).detach().numpy()
    
    eval = Evaluate(dim=(0, 1, 2))

    smape = eval.rmse_(preds, labels)
    smape = torch.tensor(smape)
    
    return smape


def masked_smape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked symmetric mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.
                                    Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = torch.where(torch.abs(labels) < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(preds-labels)/( (torch.abs(labels) + torch.abs(preds)) / 2) )
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)