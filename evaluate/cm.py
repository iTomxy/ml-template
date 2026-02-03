import numpy as np
import torch

"""
confusion matrix based metrics, suits classification and semanticsegmentation.
"""

def confusion_matrix(pred, y, num_classes, ignore_index=-1):
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
    Args:
        pred: prediction mask of shape [N] or [N, L] or [N, H, W], int
        y: ground-truth segmentation mask, same shape as pred
        num_classes: int, #classes
        ignore_index: Union[int, List[int]] = -1, class ID/s to ignore in computation
    Returns:
        tp: int[num_classes], True Positive
        tn: int[num_classes], True Negative
        fp: int[num_classes], False Positive
        fn: int[num_classes], False Negative
    """
    assert pred.dim() in [1, 2, 3]
    assert pred.shape == y.shape
    assert num_classes >= pred.max() and num_classes >= y.max()

    pred = pred.view(-1)
    y = y.view(-1)
    ignore_index = torch.tensor([ignore_index], dtype=pred.dtype).flatten().to(pred.device)
    valid_mask = ~ torch.isin(y, ignore_index)
    total_valid_pixels = valid_mask.sum().item()

    p_pred = torch.histc(pred[valid_mask], bins=num_classes, min=0, max=num_classes-1)
    p_y = torch.histc(y[valid_mask], bins=num_classes, min=0, max=num_classes-1)
    correct_mask = (pred == y) & valid_mask

    tp = torch.histc(y[correct_mask], bins=num_classes, min=0, max=num_classes-1)
    fp = p_pred - tp
    fn = p_y - tp
    tn = total_valid_pixels - tp - fp - fn

    return tp.long(), tn.long(), fp.long(), fn.long()


def clswise_cm_metrics_dist(tp, tn, fp, fn):
    """class-wise Confusion Matrix based metrics & counting
    Args:
        tp, tn, fp, fn: int[#classes], torch.LongTensor
    Returns:
        metrics: dict, {metric<str>: {
            'sum': torch.FloatTensor[#classes],
            'count': torch.FloatTensor[#classes]
        }}. Invalid entries are set to 0 so that they are directly summable.
    """
    tp = tp.to(torch.float64)
    tn = tn.to(torch.float64)
    fp = fp.to(torch.float64)
    fn = fn.to(torch.float64)
    zero = torch.zeros_like(tp, dtype=torch.float64)
    metrics = {}
    # iou
    denom = tp + fp + fn
    metrics["iou"] = {
        "sum": torch.where(denom > 0, tp / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # dice = F1
    denom = 2 * tp + fp + fn
    metrics["dice"] = {
        "sum": torch.where(denom > 0, (2 * tp) / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # sensitivity = recall
    denom = tp + fn
    metrics["sensitivity"] = {
        "sum": torch.where(denom > 0, tp / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # precision
    denom = tp + fp
    metrics["precision"] = {
        "sum": torch.where(denom > 0, tp / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # specificity
    denom = tn + fp
    metrics["specificity"] = {
        "sum": torch.where(denom > 0, tn / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # accuracy
    denom = tp + tn + fp + fn
    metrics["accuracy"] = {
        "sum": torch.where(denom > 0, (tp + tn) / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    return metrics


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
    class_set = np.arange(class_set) if isinstance(class_set, int) else np.asarray(class_set)
    mask = ~ np.isin(class_set, ignore_cls)
    metrics = {}

    # class-wise: value of all classes are kept, including those to be ignored
    metrics["iou_class"] = tp / np.clip(tp + fp + fn, 1, None)
    metrics["dice_class"] = (2 * tp) / np.clip((2 * tp + fp + fn), 1, None) # dice = F1 score
    metrics["sens_class"] = tp / np.clip(tp + fn, 1, None) # recall = sensitivity
    metrics["prec_class"] = tp / np.clip(tp + fp, 1, None)
    metrics["spec_class"] = tn / np.clip(tn + fp, 1, None) # specificity = recall for negative class
    # metrics["f1_class"] = (2 * tp) / np.clip(2 * tp + fp + fn, 1, None)
    metrics["acc_class"] = (tp + tn) / np.clip(tp + tn + fp + fn, 1, None)

    # overall average: value of ignored classes are excluded
    metrics["iou"] = float(np.mean(metrics["iou_class"][mask]))
    metrics["dice"] = float(np.mean(metrics["dice_class"][mask]))
    metrics["precision"] = float(np.mean(metrics["prec_class"][mask]))
    metrics["sensitivity"] = float(np.mean(metrics["sens_class"][mask]))
    metrics["specificity"] = float(np.mean(metrics["spec_class"][mask]))
    # metrics["f1"] = float(np.mean(metrics["f1_class"][mask]))
    metrics["acc_macro"] = float(np.mean(metrics["acc_class"][mask]))
    metrics["acc_micro"] = float((tp + tn)[mask].sum() / max(1.0, (tp + tn + fp + fn)[mask].sum()))

    for k, v in metrics.items():
        if isinstance(v, float):
            # these metrics should be within [0, 1]
            assert -0.01 < v < 1.01, "Error value range of {}: {}".format(k, v)
        else:
            # class-wise list -> convert to list for json compatibility
            metrics[k] = v.tolist()

    return metrics
