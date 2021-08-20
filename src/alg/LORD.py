import numpy as np

from alg.alg_dispatch import add_alg
from alg.gamma_scheduler import make_raw_gamma_coefs


@add_alg('LORD')
class LORD:
    """FDX controlling version of LORD 2."""
    def __init__(self,
                 delta,
                 bound,
                 startfac,
                 gamma_series,
                 alpha_strategy,
                 is_pp=True,
                 lag=0,
                 gamma_exp=None,
                 hypotheses=10000):
        self.delta = delta
        self.w0 = startfac * self.delta
        self.bound = bound
        self.is_pp = is_pp
        self.lag = lag

        # Compute the discount gamma sequence and make it sum to 1
        gamma_vec = make_raw_gamma_coefs(np.arange(1, hypotheses + 1),
                                         gamma_series, gamma_exp)
        self.gamma_vec = gamma_vec / np.sum(gamma_vec)

        if self.is_pp:
            self.first_b0 = self.delta - self.w0
            self.b0 = self.delta
        else:
            self.first_b0 = self.b0 = self.delta - self.w0

        if self.bound is not None:
            self.M_bound = self.bound - self.delta / (1 - self.delta)

        assert alpha_strategy in [2, 3]
        self.alpha_strategy = alpha_strategy

        self.wealth_vec = [self.w0]  # initial wealtself.w0
        self.alpha = []
        self.M = []
        self.rej_indices = []

    def _calc_alpha(self, idx):
        if self.alpha_strategy == 2:
            alpha = 0
            ind_rej_indices = [
                t for t in self.rej_indices if t <= len(self.alpha) - self.lag
            ]
            if ind_rej_indices:
                rej_dists = [idx - rej_idx - 1 for rej_idx in ind_rej_indices]
                alpha = np.sum(self.gamma_vec[rej_dists] * np.array(
                    [self.first_b0] +
                    [self.b0 for _ in range(len(ind_rej_indices) - 1)]))
            return alpha + self.gamma_vec[idx] * self.w0
        else:  # self.alpha_strategy == 3
            last_rej_dist = idx - self.rej_indices[
                -1] if self.rej_indices else idx + 1
            last_rej_wealth = self.wealth_vec[
                self.rej_indices[-1] + 1] if self.rej_indices else self.w0
            return self.gamma_vec[last_rej_dist - 1] * last_rej_wealth

    def _calc_cur_M(self, alpha, rej):
        return max(alpha - rej, 0)

    def _calc_cur_eta(self, alpha):
        return alpha

    def _calc_wealth(self, prev_wealth, alpha, rej):
        if not self.rej_indices:
            return prev_wealth - alpha + self.first_b0 * rej
        else:
            return prev_wealth - alpha + self.b0 * rej

    def _fill_in(self, p_len):
        fill_len = p_len - len(self.alpha)
        self.wealth_vec.extend([self.wealth_vec[-1] for _ in range(fill_len)])
        self.alpha.extend([0 for _ in range(fill_len)])
        self.M.extend([self.M[-1] if self.M else 0 for _ in range(fill_len)])

    def next_alpha(self):
        return self._calc_alpha(len(self.alpha))

    def process_next(self, p):
        idx = len(self.alpha)
        cur_alpha = self._calc_alpha(idx)
        if self.bound is not None:
            prev_M = self.M[-1] if self.M else 0
            if prev_M + self._calc_cur_eta(cur_alpha) > self.M_bound:
                cur_alpha = 0

        cur_rej = p <= cur_alpha
        if cur_rej:
            self.rej_indices.append(idx)
        self.wealth_vec.append(
            self._calc_wealth(self.wealth_vec[-1], cur_alpha, int(cur_rej)))
        if self.bound is not None:
            self.M.append(prev_M + self._calc_cur_M(cur_alpha, int(cur_rej)))
        self.alpha.append(cur_alpha)
        return cur_rej

    def run_fdr(self, pvec):
        for p in pvec:
            self.process_next(p)
        self.rejset = np.zeros(len(pvec))
        self.rejset[self.rej_indices] = 1
        return self.rejset
