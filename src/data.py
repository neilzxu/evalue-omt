from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, bernoulli, beta
from tqdm import tqdm

from alg import get_alg

_DATA_METHOD_DISPATCH = {}
DISABLE = True


def add_data_method(name):
    def add_to_dispatch(fn):
        _DATA_METHOD_DISPATCH[name] = fn
        return fn

    return add_to_dispatch


def list_data_methods():
    return _DATA_METHOD_DISPATCH.keys()


def get_data_method(name):
    return _DATA_METHOD_DISPATCH[name]


def generate_gaussian_p_values(size, non_null_p, non_null_mean, non_null_sd,
                               null_mean, null_sd):
    alternates = bernoulli.rvs(non_null_p, size=size).astype(bool)
    noise = np.random.normal(size=size)

    noise[alternates] *= non_null_sd
    noise[alternates] += non_null_mean

    noise[~alternates] *= null_sd
    noise[~alternates] += null_mean

    p_values = 1. - norm.cdf(noise, loc=null_mean, scale=null_sd)
    return p_values, alternates


@add_data_method('gaussian')
def generate_gaussian_trials(trials,
                             hypotheses,
                             non_null_p,
                             non_null_mean,
                             non_null_sd,
                             null_mean=0,
                             null_sd=1,
                             seed=None):
    """Generate trials with random p-values based on Gaussians for null and
    non-nulls.

    P-values are for a one-tailed test on whether the alternate mean is
    larger than the null mean.
    """

    if seed is not None:
        np.random.seed(seed)
    return generate_gaussian_p_values((trials, hypotheses), non_null_p,
                                      non_null_mean, non_null_sd, null_mean,
                                      null_sd)


def generate_bimodal_gaussian_trial(hypotheses, non_null_p_1, non_null_mean_1,
                                    non_null_sd_1, non_null_p_2,
                                    non_null_mean_2, non_null_sd_2, chunks,
                                    null_mean, null_sd):
    chunk_len = hypotheses // chunks
    chunk_leftover = hypotheses % chunks
    chunk_len_list = [
        chunk_len + 1 if i < chunk_leftover else chunk_len
        for i in range(chunks)
    ]

    assert sum(chunk_len_list) == hypotheses

    def param_iter():
        while True:
            yield (non_null_p_1, non_null_mean_1, non_null_sd_1)
            yield (non_null_p_2, non_null_mean_2, non_null_sd_2)

    p_value_list = []
    alternate_list = []
    for chunk_len, (non_null_p, non_null_mean,
                    non_null_sd) in zip(chunk_len_list, param_iter()):
        p_values, alternates = generate_gaussian_p_values(
            chunk_len, non_null_p, non_null_mean, non_null_sd, null_mean,
            null_sd)
        p_value_list.append(p_values)
        alternate_list.append(alternates)
    p_values, alternates = np.concatenate(p_value_list), np.concatenate(
        alternate_list)

    return p_values, alternates


def sample_localdep_gaussian(trials: int,
                             hypotheses: int,
                             max_seq_len: int,
                             mean: float,
                             sd: float,
                             rho: float,
                             neg: bool,
                             lag: int,
                             rng=None):
    rho = -rho if neg else rho
    result = np.zeros((trials * max_seq_len, hypotheses))
    if rng is None:
        noise = np.random.normal(size=(trials * max_seq_len, hypotheses)) * sd
    else:
        noise = rng.normal(size=(trials * max_seq_len, hypotheses)) * sd

    if lag > 0:
        result[:, 0] = noise[:, 0]
        init_noise_coef = np.sqrt(1 - np.square(rho))
        min_rho = np.power(rho, lag)
        prev_coef = rho / (1 - np.square(min_rho))
        noise_coef = np.sqrt(1 - (rho * prev_coef))
        assert prev_coef >= rho, (prev_coef, rho)
        assert noise_coef > 0, (noise_coef, prev_coef, min_rho, rho)
        for i in range(1, hypotheses):
            if i <= lag:
                result[:,
                       i] = rho * result[:, i - 1] + init_noise_coef * noise[:,
                                                                             i]
            else:
                corr_comp = result[:,
                                   i - 1] - (min_rho * result[:, i - lag - 1])
                result[:, i] = (prev_coef * corr_comp) + (noise_coef *
                                                          noise[:, i])
    else:
        result = noise
    return result


@add_data_method('localdep_bounded')
def generate_localdep_bounded_trials(trials,
                                     hypotheses,
                                     max_seq_len,
                                     lag,
                                     rho,
                                     null_mean,
                                     non_null_p,
                                     non_null_mean,
                                     ab_sum,
                                     bounds,
                                     neg=False,
                                     seed=None):
    """Generate trials with random p-values based for Beta RV w/ Gaussian
    copupla with local dep for null and non-nulls.

    P-values are for a one-tailed test on whether the alternate mean is
    larger than the null mean.
    """

    if seed is not None:
        np.random.seed(seed)
    noise = sample_localdep_gaussian(trials=trials,
                                     hypotheses=hypotheses,
                                     max_seq_len=max_seq_len,
                                     mean=0,
                                     sd=1,
                                     rho=rho,
                                     neg=neg,
                                     lag=lag)
    unis = norm.cdf(noise)
    unis = unis.reshape(trials, max_seq_len, hypotheses).transpose(0, 2, 1)
    alternates = bernoulli.rvs(non_null_p,
                               size=(trials, hypotheses)).astype(bool)

    alt_mask = np.tile(alternates[:, :, np.newaxis], (1, 1, max_seq_len))
    result = np.zeros(unis.shape)

    bound_range = bounds[1] - bounds[0]
    alt_beta_mean = (non_null_mean - bounds[0]) / bound_range
    result[alt_mask] = beta.ppf(unis[alt_mask],
                                a=alt_beta_mean * ab_sum,
                                b=ab_sum * (1 - alt_beta_mean),
                                loc=bounds[0],
                                scale=bound_range)

    null_beta_mean = (null_mean - bounds[0]) / bound_range
    result[~alt_mask] = beta.ppf(unis[~alt_mask],
                                 a=null_beta_mean * ab_sum,
                                 b=ab_sum * (1 - null_beta_mean),
                                 loc=bounds[0],
                                 scale=bound_range)
    #    print(f'alt mean {np.mean(result[alt_mask])}')
    return result, alternates


@add_data_method('localdep_gaussian')
def generate_localdep_gaussian_trials(trials,
                                      hypotheses,
                                      non_null_p,
                                      non_null_mean,
                                      lag,
                                      rho,
                                      max_seq_len,
                                      bounds=None,
                                      null_mean=0,
                                      sd=1,
                                      neg=False,
                                      seed=None):
    """Generate trials with random p-values based on Gaussians with local  for
    null and non-nulls.

    P-values are for a one-tailed test on whether the alternate mean is
    larger than the null mean.
    """

    # Note that subtracting the projected component of a gaussian on gaussian
    # is sufficient to make an independent component, this is how we will
    # generate the local dependence.
    #
    if seed is not None:
        np.random.seed(seed)
    noise = sample_localdep_gaussian(trials=trials,
                                     hypotheses=hypotheses,
                                     max_seq_len=max_seq_len,
                                     mean=null_mean,
                                     sd=sd,
                                     rho=rho,
                                     neg=neg,
                                     lag=lag)

    alternates = bernoulli.rvs(non_null_p,
                               size=(trials, hypotheses)).astype(bool)
    noise = noise.reshape(trials, max_seq_len, hypotheses).transpose(0, 2, 1)
    result = (non_null_mean * alternates).reshape(trials, hypotheses,
                                                  1) + noise
    return result, alternates


def create_bounded(mean, ct, bounds, ab_sum):
    ticks = np.linspace(0, 1, ct)
    bound_range = bounds[1] - bounds[0]
    beta_mean = (mean - bounds[0]) / bound_range
    values = beta.ppf(ticks,
                      a=beta_mean * ab_sum,
                      b=ab_sum * (1 - beta_mean),
                      loc=bounds[0],
                      scale=bound_range)
    # Adjust mean of pop to have same meean
    mean_diff = mean - np.mean(values)
    if mean_diff <= 0:
        sel_vec = (values > mean)
    else:
        sel_vec = (values < mean)
    inc_amount = (mean_diff * ct) / max(sel_vec.sum(), 1)
    values[sel_vec] += inc_amount
    assert np.isclose(np.mean(values),
                      mean), (np.mean(values), mean, mean_diff,
                              np.sum(sel_vec), inc_amount)
    assert np.all(values <= bounds[1]) and np.all(values >= bounds[0])
    return values


@add_data_method('discrete_bounded_wor')
def discrete_bounded_wor(null_mean: float,
                         non_null_mean: float,
                         bounds: Tuple[float, float],
                         ab_sum: float,
                         sample_size: int,
                         hypotheses: int,
                         non_null_p: float,
                         trials: int,
                         seed: Optional[float] = None) -> np.ndarray:
    if seed is not None:
        gen = np.random.default_rng(seed)
    else:
        gen = np.random.default_rng()

    null_pop, non_null_pop = create_bounded(
        null_mean, hypotheses * sample_size, bounds,
        ab_sum), create_bounded(non_null_mean, hypotheses * sample_size,
                                bounds, ab_sum)
    pop_values = np.array([null_pop, non_null_pop])
    assert np.isclose(np.mean(pop_values[0, :]), null_mean)
    assert np.isclose(np.mean(pop_values[1, :]), non_null_mean)

    null_ct = int(np.ceil((1 - non_null_p) * hypotheses))
    non_null_ct = hypotheses - null_ct
    non_null_labels = np.concatenate([np.ones(non_null_ct), np.zeros(null_ct)])

    label_list = []
    data_list = []
    # hyp_indices = np.arange(hypotheses).astype(int)
    for _ in range(trials):
        label_perm = gen.permutation(non_null_labels).astype(bool)
        label_list.append(label_perm)
        pop_val_perm = gen.permutation(pop_values,
                                       axis=1).reshape(2, hypotheses,
                                                       sample_size)
        assert np.isclose(np.mean(pop_val_perm[0, :, :]), null_mean)
        assert np.isclose(np.mean(pop_val_perm[1, :, :]), non_null_mean)
        data = np.zeros((hypotheses, sample_size))
        for i in range(hypotheses):
            if label_perm[i]:
                data[i, :] = pop_val_perm[1, i, :]
            else:
                data[i, :] = pop_val_perm[0, i, :]

        # data = pop_val_perm[label_perm.astype(int), hyp_indices, :]
        data_list.append(data)
    labels = np.stack(label_list).astype(bool)  # trials x hypotheses
    datas = np.stack(data_list)  # trials x hypotheses x sample_size
    return datas, labels
