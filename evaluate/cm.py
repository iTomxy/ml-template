import numpy as np
import torch

"""
confusion matrix based metrics, suits classification and semanticsegmentation.
"""

def confusion_matrix(pred, y, k, ignore_index=-1):
    """Compute confusion matrix (TP, TN, FP, FN) for multi-class classification/segmentation,
    based on PyTorch tensors, can be used together with `torch.distributed.all_reduce` for distributed evaluation.
    Related metrics include: IoU, dice, accuracy
    Example:
    ```python
    background_cls = 0
    for x, y in loader:
        with torch.no_grad():
            logits = model(x) # [B, C, H, W]
        pred = logits.argmax(1) # [B, H, W]
        tp, tn, fp, fn = confusion_matrix(pred, y, num_classes, background_cls)
        if ddp_enabled:
            dist.all_reduce(tp), dist.all_reduce(tn), dist.all_reduce(fp), dist.all_reduce(fn)
    ```
    Input:
        pred: prediction mask of shape [N] or [N, L] or [N, H, W], int
        y: ground-truth segmentation mask, same shape as pred
        k: int, #classes
        ignore_index: Union[int, List[int]] = -1, class ID/s to ignore in computation
    Output:
        tp: int[k], True Positive
        tn: int[k], True Negative
        fp: int[k], False Positive
        fn: int[k], False Negative
    """
    assert pred.dim() in [1, 2, 3]
    assert pred.shape == y.shape
    assert k >= pred.max() and k >= y.max()

    pred = pred.view(-1)
    y = y.view(-1)
    ignore_index = torch.tensor([ignore_index], dtype=pred.dtype).flatten().to(pred.device)
    ignore_mask = torch.isin(y, ignore_index)
    pred[ignore_mask] = -1  # set ignore_index to -1
    valid_mask = ~ ignore_mask
    total_valid_pixels = valid_mask.sum().item()

    p_pred = torch.histc(pred[valid_mask], bins=k, min=0, max=k-1)
    p_y = torch.histc(y[valid_mask], bins=k, min=0, max=k-1)
    correct_mask = (pred == y) & valid_mask

    tp = torch.histc(y[correct_mask], bins=k, min=0, max=k-1)
    fp = p_pred - tp
    fn = p_y - tp
    tn = total_valid_pixels - tp - fp - fn

    return tp.long(), tn.long(), fp.long(), fn.long()


def calc_cm_metrics(tp, tn, fp, fn, class_set, ignore_cls=[]):
    """calculate Confusion Matrix based metrics
    Input:
        tp, tn, fp, fn: int[#classes]
        class_set: int or List[int]
            - int: #classes, the class ID set will be {0, ..., n_classes - 1}
            - List[int]: ordered class ID set in the same order as tp, tn, fp & fn.
                Can be useful in part segmentation?
        ignore_cls: List[int] = [], classes to ignore at calculation, e.g. background
    Output:
        metrics: dict, {metric<str>: float}
    """
    ignore_cls = np.asarray([ignore_cls]).flatten()
    if ignore_cls.size > 0:
        class_set = np.arange(class_set) if isinstance(class_set, int) else np.asarray(class_set)
        mask = ~ np.isin(class_set, ignore_cls)
        tp, tn, fp, fn = tp[mask], tn[mask], fp[mask], fn[mask]

    metrics = {}
    metrics["iou_class"] = (tp / np.clip(tp + fp + fn, 1, None)).tolist()
    metrics["iou"] = float(np.mean(metrics["iou_class"]))
    metrics["dice_class"] = ((2 * tp) / np.clip((2 * tp + fp + fn), 1, None)).tolist()
    metrics["dice"] = float(np.mean(metrics["dice_class"]))
    metrics["prec_class"] = (tp / np.clip(tp + fp, 1, None)).tolist()
    metrics["precision"] = float(np.mean(metrics["prec_class"]))
    metrics["sens_class"] = (tp / np.clip(tp + fn, 1, None)).tolist() # recall = sensitivity
    metrics["sensitivity"] = float(np.mean(metrics["sens_class"]))
    metrics["spec_class"] = (tn / np.clip(tn + fp, 1, None)).tolist() # specificity = recall for negative class
    metrics["specificity"] = float(np.mean(metrics["spec_class"]))
    metrics["f1_class"] = ((2 * tp) / np.clip(2 * tp + fp + fn, 1, None)).tolist()
    metrics["f1"] = float(np.mean(metrics["f1_class"]))
    metrics["acc_class"] = ((tp + tn) / np.clip(tp + tn + fp + fn, 1, None)).tolist()
    metrics["acc_macro"] = float(np.mean(metrics["acc_class"]))
    metrics["acc_micro"] = float((tp + tn).sum() / max(1.0, (tp + tn + fp + fn).sum()))

    # these metrics should be within [0, 1]
    for k, v in metrics.items():
        if isinstance(v, float):
            assert -0.01 < v < 1.01, "Error value range of {}: {}".format(k, v)

    return metrics
