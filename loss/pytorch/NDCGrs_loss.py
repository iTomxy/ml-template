import torch
import torch.nn.functional as F


class NDCGrs_loss(torch.autograd.Function):
    """lower bound of tie-aware NDCG
    X: [n, bit], raw hash logit BEFORE activation like tanh/sigmoid
    L: [n, c], labels, if `sparse` then [n]
    n_bin: # of bins
    delta_scale: scaling factor for the \Delta parameter
    sparse: True if the labels are sparse class ID
    ref: https://blog.csdn.net/HackerTom/article/details/106181622
    """

    def __init__(self, n_bin, alpha=1):
        self.alpha = alpha
        self.n_bin = n_bin

    def __call__(self, X, L):
        # if n_bin is None:
        #     n_bin = bit // 2
        return NDCGrs_loss.apply(X, L, self.alpha, self.n_bin)

    @staticmethod
    def forward(ctx, X, L, alpha, n_bin, sparse=False):
        """https://github.com/kunhe/TALR/blob/master/ndcgr_s_forward.m"""
        n, bit = X.size()
        Aff = L.mm(L.T)  # [n, n]
        V = Aff.unique(sorted=True).float()  # [Naff], ascending
        Naff = V.size(0)
        Gain = 2 ** Aff - 1  # NOTE: mind overflowing, e.g., COCO dataset
        # Gns(i): gain of similarity level i
        Gns = 2 ** V - 1
        # vInd(s, i, j): whether the sim level between i-th & j-th sample is s
        S = Aff - Aff.diag().diag()
        vInd = (S.unsqueeze(2) == V.view(1, 1, -1)).float()  # [n, n, Naff]
        vInd[:, :, 0] -= vInd[:, :, 0].diag().diag()
        # for i in range(Naff):
            # print("vInd", i + 1, vInd[:, :, i].sum()) 
        Phi = 2 * torch.sigmoid(alpha * X) - 1  # [n, bit]
        Dist = (bit - Phi.mm(Phi.T)) / 2  # [n, n]
        Discount = 1 / (torch.arange(n) + 2).float().log2()
        Discount = Discount.unsqueeze(0).to(X.device)  # [1, m]
        histW = bit / n_bin
        histC = torch.linspace(0, bit, n_bin + 1).to(Dist.device)  # histogram centres
        n_hist = histC.size(0)
        # Pulse(i,j,k) > 0 means dist(i,j) lies in the region of the k-th bin
        scaled_abs_diff = (Dist.unsqueeze(2) - histC.view(1, 1, -1)).abs() / histW
        Pulse = (1 - scaled_abs_diff).clamp(min=0)  # [n, n, n_hist]
        # c_dv(i,d,v): #samples in the i-th retrieval list that
        #   of similarity level v and lie in the bin d
        c_dv = torch.zeros(n, n_hist, Naff).to(X.device)
        for _d in range(n_hist):
            for _v in range(Naff):
                _c_dv = Pulse[:, :, _d] * vInd[:, :, _v]  # [n, n]
                c_dv[:, _d, _v] = _c_dv.sum(1)  # [n]
        # c_d(i,j): #samples in the i-th retrieval list lying in j-th tie
        c_d = c_dv.sum(2)  # [n, n_hist]
        # C_d: cumsum of c_d
        C_d = c_d.cumsum(1)  # [n, n_hist]
        # C_1d: C_{d-1}
        zero = torch.zeros_like(C_d[:, 0:1])
        C_1d = torch.cat([zero, C_d[:, :-1]], 1)  # [n, n_hist]
        # C_bar = C_{d-1} + (c_d + 1) / 2 + 1
        C_bar = C_1d + (c_d + 1) / 2 + 1  # [n, n_hsit]
        # G_hat(i,j): sum gain of j-th bin in the i-th retrieval list
        G_hat = (c_dv * Gns.view(1, 1, -1)).sum(2)  # [n, n_hist]
        _DCG = (G_hat / C_bar.log2()).sum(1)  # [n]
        _DCGi = (Gain.sort(1, descending=True)[0] * Discount).sum(1)  # [n]
        # deal with invalid terms
        NDCGr_s = _DCG / _DCGi
        NDCGr_s = torch.where(torch.isinf(NDCGr_s) | torch.isnan(NDCGr_s),
                            torch.ones_like(NDCGr_s), NDCGr_s)

        ctx.save_for_backward(Phi.detach(), Dist.detach(), histC, vInd,
            torch.tensor(alpha).to(X.device), torch.tensor(histW).to(X.device),
            c_dv.detach(), C_bar.detach(), Gns, G_hat.detach(), _DCGi)
        # return (1 - NDCGr_s).sum()  # to maximize
        return NDCGr_s.mean()

    @staticmethod
    def backward(ctx, dy):
        """https://github.com/kunhe/TALR/blob/master/ndcgr_s_backward.m"""
        Phi, Dist, histC, vInd, alpha, histW, c_dv, C_bar, Gns, G_hat, _DCGi = ctx.saved_tensors
        Naff = c_dv.size(2)
        n_hist = histC.size(0)
        # 1. d(NDCGr_s)/d(c_d,v)
        d_NDCG_c = torch.zeros_like(c_dv)  # [n, n_bist, Naff]
        _one_mat = torch.ones(n_hist, n_hist).to(dy.device)
        _DCGi = _DCGi.unsqueeze(1)  # [n, 1]
        _b = G_hat / (C_bar.log2() ** 2) / C_bar / math.log(2)  # [n, n_hist]
        for _s in range(Naff):
            _t = Gns[_s] / C_bar.log2() - _b / 2 - _b.mm(_one_mat.triu(1).T)
            _t = _t / _DCGi  # [n, n_hist]
            d_NDCG_c[:, :, _s] = _t
        zero = torch.zeros_like(d_NDCG_c)
        d_NDCG_c = torch.where(
            torch.isinf(d_NDCG_c) | torch.isnan(d_NDCG_c), zero, d_NDCG_c)
        # 2. d(NDCGr_s)/d(Phi)
        d_NDCG_Phi = torch.zeros_like(Phi.T)  # [bit, n]
        _D3, _t3 = Dist.unsqueeze(2), histC.view(1, 1, -1)
        DPulse_left = ((_t3 - histW < _D3) & (_D3 <= _t3)).float()
        DPulse_right = ((_t3 < _D3) & (_D3 <= _t3 + histW)).float()
        DPulse = (DPulse_left - DPulse_right) / histW  # [n, n, n_hist]
        for _t in range(n_hist):
            dpluse = DPulse[:, :, _t]  # [n, n]
            sumA = 0
            for _s in range(Naff):
                av = d_NDCG_c[:, _t, _s].diag()
                bv = dpluse * vInd[:, :, _s]
                Av = av.mm(bv) + bv.mm(av)
                sumA = sumA + Av
            d_NDCG_Phi = d_NDCG_Phi - 0.5 * Phi.T.mm(sumA)
        d_NDCG_Phi = d_NDCG_Phi.T  # [n, bit]
        # 4. d(DCGLB)/d(x)
        sigm = (Phi + 1) / 2
        d_Phi_x = 2 * sigm * (1 - sigm) * alpha  # [n, bit]
        d_NDCG_x = - d_NDCG_Phi * d_Phi_x
        return d_NDCG_x, None, None, None, None
