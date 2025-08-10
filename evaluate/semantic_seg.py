# import packaging.version
import numpy as np
import medpy.metric.binary as mmb
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, MeanIoU, GeneralizedDiceScore, ConfusionMatrixMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
import torch

"""
Wrap segmentation metrics implemented by MedPy and MONAI in a class for easy calling.
"""

# if packaging.version.parse(np.__version__) >= packaging.version.parse('1.24'):
#     # https://github.com/loli/medpy/issues/116
#     # https://stackoverflow.com/questions/74893742/how-to-solve-attributeerror-module-numpy-has-no-attribute-bool
#     if hasattr(np, 'bool_'):
#         np.bool = np.bool_
#     else:
#         np.bool = bool

class Evaluator:
    """numpy.ndarray based segmentation evaluation for semantic segmentation."""

    METRICS = {
        "dice": mmb.dc,
        "iou": mmb.jc,
        "accuracy": lambda _B1, _B2: (_B1 == _B2).sum() / _B1.size,
        "sensitivity": mmb.sensitivity,
        "specificity": mmb.specificity,
        "hd": mmb.hd,
        "assd": mmb.assd,
        "hd95": mmb.hd95,
        "asd": mmb.asd
    }
    DISTANCE_BASED = ("hd", "assd", "hd95", "asd")

    def __init__(self, n_classes, bg_classes=[], ignore_classes=[], select=[]):
        """
        Input:
            n_classes: int, length of the softmax logit vector.
                For semantic/instance segmentation, this is the number of all classes.
                For part segmentation, this is the total number of all part categories from all object classes.
            bg_classes: int or List[int], class ID of the background class/es
                (or similar classes for all uncategorised classes).
                Typically, it is class 0.
            ignore_classes: int or List[int], ID of class/es to be ignored in evaluation.
            select: List[str], name list of metrics of interest
                Provide if you only want to evaluate on these selected metrics
                instead of all supported (see METRICS).
        """
        self.n_classes = n_classes
        if isinstance(bg_classes, int):
            bg_classes = (bg_classes,)
        self.bg_classes = bg_classes
        if isinstance(ignore_classes, int):
            ignore_classes = (ignore_classes,)
        self.ignore_classes = ignore_classes

        if len(select) == 0:
            self.metrics = self.METRICS
        else:
            self.metrics = {}
            for m in select:
                ml = m.lower()
                assert ml in self.METRICS, "Not supported metric: {}".format(m)
                self.metrics[ml] = self.METRICS[ml]

        self.reset()

    def reset(self):
        # records:
        #  - records[metr][c][i] = <metr> score of i-th datum on c-th class, or
        #  - records[metr][c] = # of NaN caused by empty pred/label
        self.records = {}
        for metr in self.metrics:
            # self.records[metr] = [[]] * self.n_classes # wrong
            self.records[metr] = [[] for _ in range(self.n_classes)]
        for metr in self.DISTANCE_BASED:
            if metr in self.metrics:
                self.records[f"empty_gt_{metr}"] = [0] * self.n_classes
                self.records[f"empty_pred_{metr}"] = [0] * self.n_classes

    def __call__(self, *, pred, y, spacing=None):
        """evaluates 1 prediction
        Input:
            pred: int numpy.ndarray, prediction (class ID after argmax) of one datum, not a batch
            y: same as `pred`, label (ground-truth class ID) of this datum
            spacing: float[] = None, len(spacing) = pred.ndim
        """
        for c in range(self.n_classes):
            B_pred_c = (pred == c).astype(np.int64)
            B_c      = (y == c).astype(np.int64)
            pred_l0, pred_inv_l0, gt_l0, gt_inv_l0 = B_pred_c.sum(), (1 - B_pred_c).sum(), B_c.sum(), (1 - B_c).sum()
            for metr, fn in self.metrics.items():
                is_distance_metr = metr in self.DISTANCE_BASED
                # if 0 == c and (self.ignore_bg or is_distance_metr):
                if c in self.ignore_classes or (is_distance_metr and c in self.bg_classes):
                    # always ignore bg for distance metrics
                    a = np.nan
                elif 0 == gt_l0 and 0 == pred_l0 and metr in ("dice", "iou", "sensitivity"):
                    a = 1
                elif 0 == gt_inv_l0 and 0 == pred_inv_l0 and "specificity" == metr:
                    a = 1
                elif is_distance_metr and pred_l0 * gt_l0 == 0: # at least one party is all 0
                    if 0 == pred_l0 and 0 == gt_l0: # both are all 0
                        # nips23a&d, xmed-lab/GenericSSL
                        a = 0
                    else: # only one party is all 0
                        a = np.nan
                        if 0 == pred_l0:
                            self.records[f"empty_pred_{metr}"][c] += 1
                        else: # 0 == gt_l0
                            self.records[f"empty_gt_{metr}"][c] += 1
                else: # normal cases or that medpy can solve well
                    # try:
                    if is_distance_metr:
                        a = fn(B_pred_c, B_c, voxelspacing=spacing)
                    else:
                        a = fn(B_pred_c, B_c)
                    # except:
                    #     a = np.nan

                self.records[metr][c].append(a)

    def load_from_dict(self, vw_dict):
        """Useful when aggregating volume-wise results to an overall one.
        Assumes the dict structure to be as follows:
        {
            "<METRIC>_cw": List[float]
            "<other keys>": Any
        }
        Only keys of format `<METRIC>_cw` are used, while other keys are ignored.
        """
        for metr in vw_dict:
            if not metr.endswith("_cw") or metr.startswith("empty_"): # only use class-wise records
                continue
            cw_list = vw_dict[metr]
            assert len(cw_list) == self.n_classes
            metr = metr[:-3] # remove "_cw"
            for c, v in enumerate(cw_list):
                if c in self.ignore_classes or (metr in self.DISTANCE_BASED and c in self.bg_classes):
                    # always ignore bg for distance metrics
                    self.records[metr][c].append(np.nan)
                else:
                    self.records[metr][c].append(v)

    def reduce(self, prec=4):
        """calculate class-wise & overall average
        Input:
            prec: int, decimal precision
        Output:
            res: dict
                - res[<metr>]: float, overall average
                - res[<metr>_cw]: List[float], class-wise average of each class
                - res[empty_pred|gt_<metr>]: int, overall #NaN caused by empty pred/label
                - res[empty_pred|gt_<metr>_cw]: List[int], class-wise #NaN
        """
        res = {}
        for metr in self.records:
            if metr.startswith("empty_"):
                res[metr+"_cw"] = self.records[metr]
                res[metr] = int(np.sum(self.records[metr]))
            else:
                CxN = np.asarray(self.records[metr], dtype=np.float32)
                nans = np.isnan(CxN)
                CxN[nans] = 0
                not_nans = ~nans

                # class-wise average
                cls_n = not_nans.sum(1) # [c]
                # cls_avg = np.where(cls_n > 0, CxN.sum(1) / cls_n, 0)
                _cls_n_denom = cls_n.copy()
                _cls_n_denom[0 == _cls_n_denom] = 1 # to avoid warning though not necessary
                cls_avg = np.where(cls_n > 0, CxN.sum(1) / _cls_n_denom, 0)
                res[f"{metr}_cw"] = np.round(cls_avg, prec).tolist()

                # overall average
                ins_cls_n = not_nans.sum(0) # [n]
                # ins_avg = np.where(ins_cls_n > 0, CxN.sum(0) / ins_cls_n, 0)
                _ins_cls_n_denom = ins_cls_n.copy()
                _ins_cls_n_denom[0 == _ins_cls_n_denom] = 1 # to avoid warning though not necessary
                ins_avg = np.where(ins_cls_n > 0, CxN.sum(0) / _ins_cls_n_denom, 0)
                ins_n = (ins_cls_n > 0).sum()
                avg = ins_avg.sum() / ins_n if ins_n > 0 else 0
                res[metr] = float(np.round(avg, prec))

        return res


class EvaluatorMonai:
    """implemented with MONAI
    Assuming `0` to be the background class.
    """

    DISTANCE_BASED = ("hd", "assd", "hd95", "asd")

    def __init__(self, n_classes, ignore_bg=False, select=[]):
        """ignore_bg: bool, for NON-distance-based metrics"""
        self.n_classes = n_classes
        self.overall_metrics, self.clswise_metrics = self.get_metrics(n_classes, not ignore_bg, select)
        self.reset()

    def get_metrics(self, n_classes, include_bg=True, select=[]):
        """instantiate MONAI segmentation measurements
        include_bg: bool, for NON-distance-based metrics
        Ref:
            - https://blog.csdn.net/HackerTom/article/details/133382705
        """
        overall_metrics = {
            "dice": DiceMetric(include_background=include_bg, reduction="mean", get_not_nans=False, ignore_empty=True, num_classes=n_classes),
            "iou": MeanIoU(include_background=include_bg, reduction="mean", get_not_nans=False, ignore_empty=True),
            "sensitivity": ConfusionMatrixMetric(include_background=include_bg, metric_name='sensitivity', reduction="mean", compute_sample=True, get_not_nans=False),
            "specificity": ConfusionMatrixMetric(include_background=include_bg, metric_name='specificity', reduction="mean", compute_sample=True, get_not_nans=False),
            "accuracy": ConfusionMatrixMetric(include_background=include_bg, metric_name='accuracy', reduction="mean", get_not_nans=False),
            # distance based metrics: always does NOT include background
            "asd": SurfaceDistanceMetric(include_background=False, symmetric=False, reduction="mean", get_not_nans=False),
            "assd": SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean", get_not_nans=False),
            "hd": HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False),
            "hd95": HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=False),
        }
        clswise_metrics = {
            "dice": DiceMetric(include_background=include_bg, reduction="mean_batch", get_not_nans=False, ignore_empty=True, num_classes=n_classes),
            "iou": MeanIoU(include_background=include_bg, reduction="mean_batch", get_not_nans=False, ignore_empty=True),
            "sensitivity": ConfusionMatrixMetric(include_background=include_bg, metric_name='sensitivity', reduction="mean_batch", compute_sample=True, get_not_nans=False),
            "specificity": ConfusionMatrixMetric(include_background=include_bg, metric_name='specificity', reduction="mean_batch", compute_sample=True, get_not_nans=False),
            "accuracy": ConfusionMatrixMetric(include_background=include_bg, metric_name='accuracy', reduction="mean_batch", get_not_nans=False),
            # distance based metrics: always does NOT include background
            "asd": SurfaceDistanceMetric(include_background=False, symmetric=False, reduction="mean_batch", get_not_nans=False),
            "assd": SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean_batch", get_not_nans=False),
            "hd": HausdorffDistanceMetric(include_background=False, reduction="mean_batch", get_not_nans=False),
            "hd95": HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False),
        }
        if len(select) > 0:
            overall_metrics = {k: v for k, v in overall_metrics.items() if k in select}
            clswise_metrics = {k: v for k, v in clswise_metrics.items() if k in select}

        return overall_metrics, clswise_metrics

    def reset(self):
        for k in self.overall_metrics:
            self.overall_metrics[k].reset()
        for k in self.clswise_metrics:
            self.clswise_metrics[k].reset()

    def __call__(self, *, pred, y, spacing=None):
        """evaluates 1 prediction
        Input:
            pred: torch.Tensor, typically [H, W] or [H, W, L], predicted class ID
            y: same as `pred`, label, ground-truth class ID
            spacing: float[] = None, len(spacing) = pred.ndim
        """
        if pred.dim() == 1:
            # NOTE Vectors of shape [L] are NOT natually supported (by distance-based metrics?).
            # I guess this is because it does not form a object surface.
            pred, y = pred.unsqueeze(1), y.unsqueeze(1) # [L] -> [L, 1], pretending [H, W]

        pred = one_hot(pred.unsqueeze(0).unsqueeze(0), num_classes=self.n_classes, dim=1) # -> (B=1, C, H, W[, L])
        y = one_hot(y.unsqueeze(0).unsqueeze(0), num_classes=self.n_classes, dim=1) # -> (B=1, C, H, W[, L])
        for k in self.overall_metrics:
            if k in self.DISTANCE_BASED:
                self.overall_metrics[k](y_pred=pred, y=y, spacing=spacing)
            else:
                self.overall_metrics[k](y_pred=pred, y=y)

        for k in self.clswise_metrics:
            if k in self.DISTANCE_BASED:
                self.clswise_metrics[k](y_pred=pred, y=y, spacing=spacing)
            else:
                self.clswise_metrics[k](y_pred=pred, y=y)

    def reduce(self, prec=4):
        res = {}
        for k in self.overall_metrics:
            # try:
            res[k] = self.metric_get_off(self.overall_metrics[k].aggregate(), prec)
            # except:
            #     print(k, type(self.overall_metrics[k]))

        for k in self.clswise_metrics:
            r = self.metric_get_off(self.clswise_metrics[k].aggregate(), prec)
            # Ensure the class-wise metrics are in list format
            # to be consistent with `Evaluator` above.
            if isinstance(r, (float, int)):
                r = [r]

            # In case of background(0) is excluded,
            # prepend a `0` to ensure the the length equals to #classes.
            # This assumes `0` is the background class.
            if len(r) != self.n_classes:
                assert len(r) == self.n_classes - 1
                r = [0] + r

            res[k+"_cw"] = r

        return res

    def metric_get_off(self, res, prec=6):
        """convert MONAI evaluation results to JSON-serializable format"""
        if isinstance(res, list):
            assert len(res) == 1
            res = res[0]

        if "cuda" in res.device.type:
            res = res.cpu()

        if 0 == res.ndim:
            res = round(res.item(), prec)
        else:
            res = list(map(lambda x: round(x, prec), res.tolist()))
            if len(res) == 1:
                res = res[0]

        return res


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
