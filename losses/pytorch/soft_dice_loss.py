from collections import defaultdict

import torch
from torch.nn import functional as F

class SoftDiceLoss:
    def __init__(self, device, dice_loss_weights=None, n_classes=2):
        assert (
            n_classes > 1
        ), "Even if it is a binary classification, please use 2 classes instead of one"
        self.dice_loss_weights = dice_loss_weights
        self.n_classes = n_classes
        self.device = device

    @property
    def unreduced_loss(self):
        return self._unreduced_loss

    @property
    def per_sample_loss(self):
        return self._per_sample_loss

    @property
    def reduced_loss(self):
        return self._reduced_loss

    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        loss, individual_losses = dice_loss(
            prediction,
            target,
            self.dice_loss_weights,
            self.n_classes,
            self.device,
            return_individual_losses=True,
        )
        self._unreduced_loss = loss
        self._per_sample_loss = loss
        self._reduced_loss = loss.mean()

        return self._reduced_loss


def dice_loss(
    preds, targets, dice_loss_weights=None, n_classes=2, device=None, return_individual_losses=False
):
    def _set_all_label_weights_to_one():
        return {f"weight_for_class_{i}": 1 for i in range(n_classes)}

    if dice_loss_weights is None:
        dice_loss_weights = _set_all_label_weights_to_one()

    dice_loss_ = multi_class_dice_loss(preds, targets.long(), n_classes=n_classes, device=device)
    
    dice_loss_lookup = {}

    for k, v in dice_loss_.items():
        class_label = k
        dice_loss_lookup[k] = v * dice_loss_weights[f"weight_for_class_{class_label}"]
        
    loss = torch.stack([loss for loss in dice_loss_lookup.values()]).mean(dim=0)
    
    if return_individual_losses:
        return loss, dice_loss_lookup
    else:
        return loss


def multi_class_dice_loss(pred_logits, targets, n_classes, device):
    """

    :param pred_logits: B x C x H x W
    :param targets: B x 1 x H x W, type Long
    :return:
    """

    one_hot = torch.FloatTensor(
        targets.size(0), n_classes, targets.size(2), targets.size(3)
    ).zero_()
    target_one_hot = one_hot.scatter_(1, targets.cpu().data, 1).to(device)
    pred_probs = F.softmax(pred_logits, dim=1)
    batch_size = pred_logits.size(0)
    loss_dict = defaultdict(list)

    for batch_idx in range(batch_size):
        for class_i in range(n_classes):
            loss_for_class_i = single_class_dice_loss(
                pred_probs[batch_idx, class_i, :, :],
                target_one_hot[batch_idx, class_i, :, :],
            )
            loss_dict[class_i].append(loss_for_class_i)

    return {k: torch.stack(v) for k, v in loss_dict.items()}


def single_class_dice_loss(inputs, one_hots):
    inputs_flat = inputs.contiguous().view(-1)
    one_hots_flat = one_hots.contiguous().view(-1)
    intersection = torch.sum(inputs_flat * one_hots_flat)
    
    eps = 1e-8
    return 1 - (
        (2 * intersection + eps) / (torch.sum(inputs_flat) + torch.sum(one_hots_flat) + eps)
    )