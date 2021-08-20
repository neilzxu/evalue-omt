from typing import Optional

import numpy as np

from alg.alg_dispatch import add_alg
from alg.gamma_scheduler import make_raw_gamma_coefs


def ell(t: int):
    """Compute tth harmonic number."""
    return np.sum(1 / np.arange(1, t + 1))


def beta_fn(reshaping_method, t, r):
    if reshaping_method == 'BY':
        level = np.minimum(np.floor(r), t) / ell(t)
        return level
    else:
        assert False


@add_alg('LOND')
class LOND:
    """LOND algorithm.

    Subsumes e-LOND, since e-LOND simply assume the p-vaues are inverses
    of e-values and the reshaping does not need to be used even under
    arbitrary dependence.
    """
    def __init__(self,
                 delta: float,
                 gamma_series: str,
                 reshaping: Optional[str] = None,
                 gamma_exp: Optional[float] = None,
                 rand: Optional[str] = None,
                 seed: Optional[int] = None,
                 hypotheses: int = 10000):
        self.delta = delta

        # Compute the discount gamma sequence and make it sum to 1
        gamma_vec = make_raw_gamma_coefs(np.arange(1, hypotheses + 1),
                                         gamma_series, gamma_exp)
        self.gamma_vec = np.append(0, gamma_vec / np.sum(gamma_vec))
        self.reshaping = reshaping
        self.alpha = []
        self.rej_indices = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.rand = rand
        if self.rand == 'single':
            self.U = self.rng.uniform(0, 1)

    def _next_alpha(self):
        t = len(self.alpha) + 1

        rejs = len(self.rej_indices) + 1

        if self.rand is not None:
            U = self.rng.uniform(0, 1) if self.rand == 'ind' else self.U
        else:
            U = 1

        if self.reshaping is None:
            alpha = self.delta * self.gamma_vec[t] * rejs / U
        else:
            alpha = self.delta * self.gamma_vec[t] * beta_fn(
                self.reshaping, t, rejs / U)
        return alpha

    def process_next(self, p):
        idx = len(self.alpha)
        cur_alpha = self._next_alpha()
        # print(p, cur_alpha)
        cur_rej = p <= cur_alpha
        if cur_rej:
            self.rej_indices.append(idx)
        self.alpha.append(cur_alpha)
        return cur_rej

    def run_fdr(self, pvec):
        for p in pvec:
            self.process_next(p)
        self.rejset = np.zeros(len(pvec))
        self.rejset[self.rej_indices] = 1
        if self.reshaping is None:
            guar_rejs = pvec <= (self.delta * self.gamma_vec[1:])
            assert np.all(self.rejset[guar_rejs])

        return self.rejset
