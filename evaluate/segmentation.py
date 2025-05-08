import numpy as np
import medpy.metric.binary as mmb

"""
Wrap segmentation metrics implemented by MedPy in a class for easy calling.
"""

class SemanticSegEvaluator:
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

    def __init__(self, n_classes, bg_classes, ignore_classes=[], select=[]):
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

        self.metrics = {}
        _no_select = len(select) == 0
        for m, f in self.METRICS.items():
            if _no_select or m in select:
                self.metrics[m] = f

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

    def __call__(self, *, pred, y):
        """evaluates 1 prediction
        Input:
            pred: int numpy.ndarray, prediction (class ID after argmax) of one datum, not a batch
            y: same as `pred`, label (ground-truth class ID) of this datum
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
                    a = fn(B_pred_c, B_c)
                    # except:
                    #     a = np.nan

                self.records[metr][c].append(a)

    def reduce(self, prec=4):
        """calculate class-wise & overall average
        Input:
            prec: int, decimal precision
        Output:
            res: dict
                - res[<metr>]: float, overall average
                - res[<metr>_clswise]: List[float], class-wise average of each class
                - res[empty_pred|gt_<metr>]: int, overall #NaN caused by empty pred/label
                - res[empty_pred|gt_<metr>_clswise]: List[int], class-wise #NaN
        """
        res = {}
        for metr in self.records:
            if metr.startswith("empty_"):
                res[metr+"_clswise"] = self.records[metr]
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
                res[f"{metr}_clswise"] = np.round(cls_avg, prec).tolist()

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
