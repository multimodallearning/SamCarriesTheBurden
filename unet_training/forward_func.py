from typing import List

import torch
from clearml import Logger
from torch import functional as F
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from utils.dice_coefficient import multilabel_dice


def forward_bce(mode: str, data_loader: DataLoader, epoch: int,  # have to be given each call
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

        with torch.no_grad():
            # normalize
            x = (x - data_loader.dataset.IMG_MEAN) / data_loader.dataset.IMG_STD
            if data_aug > 0:  # apply data augmentation
                theta = torch.eye(2, 3, device=device).unsqueeze(0) + torch.randn(len(x), 2, 3,
                                                                                  device=device) * data_aug
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


def inverse_theta(theta: torch.Tensor) -> torch.Tensor:
    identity_vec = torch.tensor([0, 0, 1]).view(1, 1, 3).expand(len(theta), -1, -1).to(theta)
    theta_squared = torch.cat([theta, identity_vec], dim=1)
    theta_inv = theta_squared.inverse()
    theta_inv = theta_inv[:, :2, :]
    return theta_inv


def forward_mean_teacher_bce(mode: str, data_loader: DataLoader, epoch: int,  # have to be given each call
                             # can be provided via kwargs dict
                             student: nn.Module, teacher: nn.Module, optimizer: Optimizer, device: torch.device,
                             bce_pos_weight: torch.Tensor, alpha: float, loss_collectors: List[MeanMetric],
                             data_aug: float = 0) -> (torch.Tensor, torch.Tensor):
    # set model mode according to mode
    if mode == 'train':
        student.train()
        teacher.train()
    elif mode in ['test', 'val']:
        student.eval()
        teacher.eval()
        data_aug = 0  # disable data augmentation during testing
    else:
        raise ValueError(f'Unknown mode: {mode}')
    assert student.training == teacher.training, 'Student and teacher must have the same mode'

    # reset loss collectors
    for lc in loss_collectors:
        lc.reset()
    student_loss_collector, teacher_loss_collector, consistency_loss_collector = loss_collectors
    dsc_s = []
    dsc_t = []

    for batch in data_loader:
        if student.training:  # training with mixed labeled and unlabeled data
            x, y, _, gt_mask = batch
            x, gt_mask = x.to(device, non_blocking=True), gt_mask.to(device, non_blocking=True)
            if y is not None:
                y = y.to(device, non_blocking=True)
        else:  # validation or testing with only labeled data
            x, y, _ = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            gt_mask = torch.ones(len(x), dtype=torch.bool, device=device)

        with torch.no_grad():
            # normalize
            x = (x - data_loader.dataset.IMG_MEAN) / data_loader.dataset.IMG_STD
            if data_aug > 0:  # apply data augmentation
                identity = torch.eye(2, 3, device=device).unsqueeze(0)
                theta_s = identity + torch.randn(len(x), 2, 3, device=device) * data_aug
                theta_t = identity + torch.randn(len(x), 2, 3, device=device) * data_aug

                theta_s_inv = inverse_theta(theta_s)
                theta_t_inv = inverse_theta(theta_t)

                x_s = F.grid_sample(x, F.affine_grid(theta_s, x.shape, align_corners=False), align_corners=False)
                x_t = F.grid_sample(x, F.affine_grid(theta_t, x.shape, align_corners=False), align_corners=False)
            else:
                x_s = x_t = x
                if gt_mask.any():
                    theta_s_inv = theta_t_inv = torch.eye(2, 3, device=device).unsqueeze(0).expand(len(y), -1, -1)
            del x

            # teacher forward
            y_hat_t = teacher(x_t)
            y_hat_t = F.grid_sample(y_hat_t, F.affine_grid(theta_t_inv, y_hat_t.shape, align_corners=False),
                                    align_corners=False)

            # classification loss (only for monitoring)
            if gt_mask.any():
                teacher_loss = F.binary_cross_entropy_with_logits(y_hat_t[gt_mask], y, pos_weight=bce_pos_weight)
                teacher_loss_collector(teacher_loss)

                dsc_t.append(multilabel_dice(y_hat_t[gt_mask].sigmoid() > 0.5, y.bool()))

        # student
        with torch.set_grad_enabled(student.training):  # forward
            y_hat_s = student(x_s)
            y_hat_s = F.grid_sample(y_hat_s, F.affine_grid(theta_s_inv, y_hat_s.shape, align_corners=False),
                                    align_corners=False)

            # classification loss
            if gt_mask.any():
                student_loss = F.binary_cross_entropy_with_logits(y_hat_s[gt_mask], y, pos_weight=bce_pos_weight)
                student_loss_collector(student_loss)

                dsc_s.append(multilabel_dice(y_hat_s[gt_mask].sigmoid() > 0.5, y.bool()))
            else:
                student_loss = 0

            # consistency loss
            consistency_loss = F.mse_loss(y_hat_s, y_hat_t)
            consistency_loss_collector(consistency_loss)

            loss = student_loss + consistency_loss

        if student.training:  # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # teacher update
            with torch.no_grad():
                for ema_param, param in zip(teacher.parameters(), student.parameters()):
                    ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    # log metrics scalars
    dsc_s = torch.cat(dsc_s, dim=0)
    dsc_t = torch.cat(dsc_t, dim=0)
    log = Logger.current_logger()
    log.report_scalar('Consistency', mode, iteration=epoch, value=consistency_loss_collector.compute().item())
    log.report_scalar('BCE ' + mode, 'teacher', iteration=epoch, value=teacher_loss_collector.compute().item())
    log.report_scalar('BCE ' + mode, 'student', iteration=epoch, value=student_loss_collector.compute().item())
    log.report_scalar('Dice ' + mode, 'teacher', iteration=epoch, value=dsc_t.nanmean().item())
    log.report_scalar('Dice ' + mode, 'student', iteration=epoch, value=dsc_s.nanmean().item())

    log.report_histogram('Dice ' + mode, 'teacher', iteration=epoch,
                         values=dsc_t.nanmean(0).cpu().numpy(),
                         xlabels=data_loader.dataset.BONE_LABEL, xaxis='class', yaxis='dice')

    log.report_histogram('Dice ' + mode, 'student', iteration=epoch,
                         values=dsc_s.nanmean(0).cpu().numpy(),
                         xlabels=data_loader.dataset.BONE_LABEL, xaxis='class', yaxis='dice')
