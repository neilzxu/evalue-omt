from typing import Union
from methodtools import lru_cache

import numpy as np

numtype = Union[int, float, np.ndarray]


def make_raw_gamma_coefs(idxs, series, exponent=None):
    if series == 'gaussian':  # asymptotically optimal for gaussian
        numerator = np.log(np.maximum(idxs, 2))
        denominator = idxs * np.exp(np.sqrt(np.log(np.maximum(idxs, 1))))
        gamma_vec = numerator / denominator
    elif series == 'beta':  # asymptotically optimal for beta
        gamma_vec = np.power(
            np.log(np.maximum(idxs, 2 * np.ones(len(idxs)))) / idxs, 0.5)
    elif series == "easy":
        gamma_vec = 1 / ((idxs + 1) * (idxs + 2))
    elif series == "uniform":
        gamma_vec = np.ones(idxs.shape)
    else:  # series == 'inverse'
        assert exponent is not None
        gamma_vec = 1. / (idxs**(exponent))
    return gamma_vec


def make_raw_acc_coef(idxs, series, wealth, w0, decay_coef, exponent=None):
    gamma_coefs = make_raw_gamma_coefs(idxs=idxs,
                                       series=series,
                                       exponent=exponent)
    return np.power(gamma_coefs, wealth / w0 * decay_coef)


def make_cached_adaptive_gammas(gamma_coefs):
    @lru_cache(maxsize=None)
    def get_normalizer(decay_len, wealth, w0, decay_coef):
        raw_coefs = np.power(gamma_coefs[:decay_len], wealth / w0 * decay_coef)
        return np.sum(raw_coefs)

    def get_adaptive_gammas(idxs: np.ndarray, wealth: np.ndarray, w0: numtype,
                            decay_coef: numtype, decay_len: int):
        assert decay_len <= len(gamma_coefs)
        np.power(gamma_coefs[idxs], wealth / w0 * decay_coef) / np.array(
            [get_normalizer(decay_len, w, w0, decay_coef) for w in wealth])

    return get_adaptive_gammas


class GammaScheduler:
    def __init__(self,
                 mode,
                 series,
                 decay_len: Union[None, int] = None,
                 decay_coef: Union[None, float, int] = None,
                 exponent: Union[None, float] = None):

        assert series in ['gaussian', 'beta', 'inverse']
        self.series = series
        assert mode in ['static', 'dynamic', 'dynamic_uni']
        self.mode = mode

        if series == 'inverse':
            assert exponent is not None and isinstance(decay_len, float)
        self.exponent = exponent

        self.gamma_coefs = make_raw_gamma_coefs(np.arange(1, 200000), series,
                                                self.exponent)
        self.gamma_coefs /= np.sum(self.gamma_coefs)

        self.decay_len = decay_len
        self.decay_coef = decay_coef
        if mode == 'dynamic':
            assert decay_len is not None and isinstance(decay_len, int)
            assert decay_coef is not None and (isinstance(decay_coef, float)
                                               or isinstance(decay_coef, int))
        self.normalizer_map = {}

    def __getstate__(self):
        self.normalizer_map = {}
        state = self.__dict__.copy()
        return state

    def get_normalizer(self, wealth, w0):
        if (wealth, w0) in self.normalizer_map:
            return self.normalizer_map[wealth, w0]
        normalizer = np.sum(
            np.power(self.gamma_coefs[:self.decay_len],
                     wealth / w0 * self.decay_coef))
        self.normalizer_map[wealth, w0] = normalizer
        return normalizer

    def get_adaptive_gammas(self, idxs: np.ndarray, wealth: np.ndarray,
                            w0: numtype):
        return np.power(self.gamma_coefs[idxs],
                        wealth / w0 * self.decay_coef) / np.array(
                            [self.get_normalizer(w, w0) for w in wealth])

    def make_gammas(self, dists, wealth_vec=None, w0=None):
        idxs = dists - 1
        if self.mode == 'static':
            return self.gamma_coefs[idxs]
        else:  # self.mode in ["dynamic", "dynamic_uni"]
            assert wealth_vec is not None and w0 is not None
            assert w0 > 0

            gamma_vec = self.gamma_coefs[idxs]

            big_wealth_mask = (wealth_vec / w0 * self.decay_coef) > 1.
            past_decay_len_mask = idxs < self.decay_len
            adaptive_mask = big_wealth_mask & past_decay_len_mask
            adaptive_expired_mask = big_wealth_mask & ~past_decay_len_mask

            gamma_vec[adaptive_expired_mask] = 0.
            if self.mode == 'dynamic':
                adaptive_gammas = self.get_adaptive_gammas(
                    idxs[adaptive_mask], wealth_vec[adaptive_mask], w0)
            else:  # self.mode == 'dynamic_uni'
                adaptive_gammas = np.ones(
                    shape=np.sum(adaptive_mask)) / self.decay_len
            gamma_vec[adaptive_mask] = adaptive_gammas
            return gamma_vec
