import torch
from clearml import Logger
from kornia.augmentation.container import AugmentationSequential
from kornia.enhance import Normalize
from torch import functional as F
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from utils.dice_coefficient import multilabel_dice


def forward_bce(mode: str, data_loader: DataLoader, epoch: int,  # have to given each call
                # can be provided via kwargs dict
                model: nn.Module, optimizer: Optimizer, device: torch.device, bce_pos_weight: torch.Tensor,
                loss_collector: MeanMetric, data_aug: float = 0) -> (torch.Tensor, torch.Tensor):
    # set model mode according to mode
    if mode == 'train':
        model.train()
    elif mode in ['test', 'val']:
        model.eval()
        data_aug = 0  # disable data augmentation during testing
    else:
        raise ValueError(f'Unknown mode: {mode}')

    loss_collector.reset()
    dsc = []

    for x, y, _ in data_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if data_aug > 0:  # apply data augmentation
            with torch.no_grad():
                theta = torch.eye(2, 3, device=device).unsqueeze(0) + torch.randn(len(x), 2, 3, device=device) * data_aug
                affine = F.affine_grid(theta, x.shape, align_corners=False)
                x = F.grid_sample(x, affine, align_corners=False)
                y = F.grid_sample(y, affine, align_corners=False, mode='nearest')

        with torch.set_grad_enabled(model.training):  # forward
            y_hat = model(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=bce_pos_weight)

        if model.training:  # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # track metrics
        loss_collector(loss)
        dsc.append(multilabel_dice(y_hat.sigmoid() > 0.5, y.bool()))

    # log metrics scalars
    dsc = torch.cat(dsc, dim=0)
    log = Logger.current_logger()
    log.report_scalar('BCE', mode, iteration=epoch, value=loss_collector.compute().item())
    log.report_scalar('Dice', mode, iteration=epoch, value=dsc.nanmean().item())

    log.report_histogram('Dice', mode, iteration=epoch,
                         values=dsc.nanmean(0).cpu().numpy(),
                         xlabels=data_loader.dataset.BONE_LABEL, xaxis='class', yaxis='dice')

    return dsc.nanmean(), loss_collector.compute()
