import torch


@torch.inference_mode()
def multiclass_dice(y_hat: torch.Tensor, y: torch.Tensor, max_label: int) -> torch.Tensor:
    """
    Compute the Dice Coefficient for each class. Class 0 is considered background and is ignored.
    :param y_hat: integer map of predicted labels (B, *)
    :param y: integer map of ground truth labels (B, *)
    :param max_label: maximum label value
    :return: dice coefficient for each class (B, max_label)
    """
    assert y_hat.shape == y.shape, f'Shape mismatch: {y_hat.shape} != {y.shape}'
    assert y_hat.dtype == y.dtype == torch.long
    assert y_hat.ndim > 1

    B = y.shape[0]
    y = y.view(B, -1)
    y_hat = y_hat.view(B, -1)

    tmp = torch.arange(1, max_label + 1, device=y.device, dtype=torch.long).view(1, -1, 1)
    y_hat = y_hat.unsqueeze(1) == tmp
    y = y.unsqueeze(1) == tmp

    dsc = multilabel_dice(y_hat, y)
    return dsc


@torch.inference_mode()
def multilabel_dice(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Dice Coefficient for each class.
    :param y_hat: prediction tensor of shape (B, C, *)
    :param y: ground truth tensor of shape (B, C, *)
    :return: dice coefficient for each class (B, C)
    """
    assert y_hat.shape == y.shape, f'Shape mismatch: {y_hat.shape} != {y.shape}'
    assert y_hat.dtype == y.dtype == torch.bool
    assert y_hat.ndim > 2

    B, C = y.shape[:2]
    y_hat = y_hat.view(B, C, -1).float()
    y = y.view(B, C, -1).float()

    # calculate dice
    intersection = torch.sum(y_hat * y, dim=2)
    cardinality = torch.sum(y_hat + y, dim=2)
    dice = 2 * intersection / (cardinality + 1e-8)

    # mark invalid calculations
    dice[y.any(2).logical_not()] = float('nan')

    return dice
