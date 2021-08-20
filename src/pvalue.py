from typing import List, Optional, Tuple

from confseq.betting import betting_mart
from confseq.betting_strategies import lambda_aKelly
from itertools import product
import obp
import numpy as np
import scipy.special
import scipy.stats

from alg.gamma_scheduler import make_raw_gamma_coefs

_PVAL_DISPATCH = {}
_POLICY_DISPATCH = {}


def get_pvalue(name):
    return _PVAL_DISPATCH[name]


def get_policy(name):
    return _POLICY_DISPATCH[name]


def pvalue(name):
    def add_to_dispatch(fn):
        _PVAL_DISPATCH[name] = fn
        return fn

    return add_to_dispatch


def policy(name):
    def add_to_dispatch(fn):
        _POLICY_DISPATCH[name] = fn
        return fn

    return add_to_dispatch


@pvalue('e_gaussian_nsm')
def e_gaussian_nsm(x: np.ndarray, est_mu=3, var=1):
    """Gaussian NSM.

    :param x: max_seq_len length array

    :return: running minimum of Gaussian NSM
    """
    ps = []
    for run in x:
        p = np.exp(np.cumsum(-1 * (est_mu * x -
                                   (var * np.square(est_mu) / 2))))
        ps.append(np.min(p))
    return ps


@pvalue('max_p')
def max_p(x, pvalue_spec, pvalue_max_spec):
    pvalues = get_pvalue(pvalue_spec['method'])(x.copy(),
                                                **(pvalue_spec['kwargs']))
    pvalues_min = get_pvalue(pvalue_max_spec['method'])(
        x.copy(), **(pvalue_max_spec['kwargs']))
    return np.maximum(pvalues, pvalues_min)


def _evalue_stop(version, log_e, alpha=None, gammas=None):
    """Return e-value at certain types of stopping rules."""

    log_e_max = log_e.max(axis=2)
    log_e_final = log_e[:, :, -1]
    if version == 'inf':
        log_e_res = log_e_max
        res = np.exp(-1 * np.maximum(0, log_e_res))
    elif version == 'mart_stopped':
        # *Summary*: stopping rule that can be used where samples across
        # hypotheses are dependent, but still martingale/independent across
        # max_seq_len.
        # Use the information from samples <= t and hypotheses <= i to
        # determine rejections that can be confirmed so far by a
        # sample size.
        rej_so_far = np.zeros((trials, max_seq_len))
        log_e_res = np.zeros((trials, hypotheses))
        assert not np.any(np.isnan(log_e)), f'log_e ({log_e.shape}): {log_e}'
        for i in range(hypotheses):
            cum_rej_so_far = np.cumsum(rej_so_far, axis=-1)
            rej_level = np.log(1 / (alpha * norm_gammas[i] *
                                    (cum_rej_so_far + 1)))
            assert not np.any(np.isnan(rej_level)), f'rej_level: {rej_level}'
            rej_conds = log_e[:, i, :] >= rej_level
            new_rejs = np.any(rej_conds, axis=-1)
            sample_rej_idxs = np.argmax(rej_conds[np.where(new_rejs)], axis=-1)
            # set result for new rejections to be at threshold, and final value
            # for those who didn't reject
            log_e_res[np.where(new_rejs), i] = log_e[np.where(new_rejs), i,
                                                     sample_rej_idxs]
            log_e_res[np.where(~new_rejs), i] = log_e[np.where(~new_rejs), i,
                                                      -1]
            # update array of rejections made before a sample idx
            rej_so_far[np.where(new_rejs), sample_rej_idxs] += 1
        res = np.exp(-1 * np.maximum(0, log_e_res))
    # elif version == 'stopped':
    #     log_e_res = np.zeros((trials, hypotheses))
    #     threshold = 1 / (norm_gammas * alpha)
    #     stop_cond = log_e_max >= threshold
    #     sample_stop_idxs = np.argmax(stop_cond[stop_cond], axis=-1)
    #     res = stop_cond.astype(float) * np.exp(-1 * np.maximum(
    #         0, log_e_max)) + (~stop_cond).astype(float) * np.exp(
    #             -1 * np.maximum(0, log_e_final))
    else:
        assert version == 'final'
        log_e_res = log_e_final
        res = np.exp(-1 * np.maximum(0, log_e_res))
    return res


@pvalue('e_gaussian_mgf')
def e_gaussian_mgf(x,
                   est_mu=3,
                   var=1,
                   null=0,
                   alpha=0.05,
                   alpha_guess='expect',
                   version='final',
                   gamma_series='easy',
                   gamma_exp=None):
    """x.shape == (trials, hypotheses, max_seq_len)

    We take the running minimum, b/c we assume the process will stop
    after hitting the test threshold which is the same as testing if the
    process min is smaller than a threshold.
    """
    trials, hypotheses, max_seq_len = x.shape
    raw_gammas = make_raw_gamma_coefs(np.arange(1, hypotheses + 1),
                                      series=gamma_series,
                                      exponent=gamma_exp)
    norm_gammas = raw_gammas / np.sum(raw_gammas)
    min_alpha_levels = alpha * norm_gammas
    if alpha_guess == 'expect':
        # Calculate, if all p-values were uniform, the expected rej at t
        expected_rej = np.zeros(hypotheses)
        for i in range(hypotheses):
            if i == 0:
                expected_rej[i] = alpha * norm_gammas[i]
            else:
                expected_rej[i] = alpha * norm_gammas[i] * (
                    expected_rej[i - 1] + 1) + expected_rej[i - 1]
        guess_alpha_levels = alpha * norm_gammas * expected_rej
    else:  # alpha_guess == 'min'
        guess_alpha_levels = min_alpha_levels

    lam = np.tile(np.sqrt(2 * np.log(1 / guess_alpha_levels) / max_seq_len),
                  (trials, 1))[:, :, np.newaxis]
    assert not np.any(np.isnan(lam)), f'lam: {lam}'

    log_e = (lam * (x - null) - (var * np.square(lam) / 2)).cumsum(axis=2)

    # betting_lam = np.minimum(lam, np.abs(bounds[0] - null) / 2)
    # betting_log_e = np.log(1 + lam * (x - null)).cumsum(axis=2)
    log_e_max = log_e.max(axis=2)
    log_e_final = log_e[:, :, -1]
    if version == 'inf':
        log_e_res = log_e_max
        res = np.exp(-1 * np.maximum(0, log_e_res))
    elif version == 'mart_stopped':
        # *Summary*: stopping rule that can be used where samples across
        # hypotheses are dependent, but still martingale/independent across
        # max_seq_len.
        # Use the information from samples <= t and hypotheses <= i to
        # determine rejections that can be confirmed so far by a
        # sample size.
        rej_so_far = np.zeros((trials, max_seq_len))
        log_e_res = np.zeros((trials, hypotheses))
        assert not np.any(np.isnan(log_e)), f'log_e ({log_e.shape}): {log_e}'
        for i in range(hypotheses):
            cum_rej_so_far = np.cumsum(rej_so_far, axis=-1)
            rej_level = np.log(1 / (alpha * norm_gammas[i] *
                                    (cum_rej_so_far + 1)))
            assert not np.any(np.isnan(rej_level)), f'rej_level: {rej_level}'
            rej_conds = log_e[:, i, :] >= rej_level
            new_rejs = np.any(rej_conds, axis=-1)
            sample_rej_idxs = np.argmax(rej_conds[np.where(new_rejs)], axis=-1)
            # set result for new rejections to be at threshold, and final value
            # for those who didn't reject
            log_e_res[np.where(new_rejs), i] = log_e[np.where(new_rejs), i,
                                                     sample_rej_idxs]
            log_e_res[np.where(~new_rejs), i] = log_e[np.where(~new_rejs), i,
                                                      -1]
            # update array of rejections made before a sample idx
            rej_so_far[np.where(new_rejs), sample_rej_idxs] += 1
        # don't take max for ones that go into e-value algs b/c of
        # randomization techniques
        res = np.exp(-1 * log_e_res)
    elif version == 'stopped':
        threshold = 1 / (norm_gammas * alpha)
        stop_cond = log_e_max >= threshold
        res = stop_cond.astype(float) * np.tile(
            min_alpha_levels,
            (trials, 1)) + (~stop_cond).astype(float) * np.exp(
                -1 * np.maximum(0, log_e_final))
    else:
        log_e_res = log_e_final
        res = np.exp(-1 * np.maximum(0, log_e_res))

    assert not np.any(
        np.isnan(res)
    ), f'{np.sum(np.isnan(res))}, {res.shape}, {version}, {log_e_res[np.isnan(res)]}'
    return res


@pvalue('gaussian_cdf')
def gaussian_cdf(x, null_mean=0, var=1, one_sided=True):
    if len(x.shape) == 3:
        seq_len = x.shape[2]
        total_var = var * seq_len
        total_null_mean = null_mean * seq_len
        x = x.sum(axis=2)
    else:
        total_null_mean = null_mean
        total_var = var

    sd = np.sqrt(total_var)
    if one_sided:
        return 1. - scipy.stats.norm.cdf(x, loc=total_null_mean, scale=sd)
    else:
        res = scipy.stats.norm.cdf(x, loc=total_null_mean, scale=sd)
        return 2. * np.minimum(res, 1. - res)


@policy('random_gaussian_policy')
def random_gaussian_policy(seed, epsilon, **kwargs):
    rng = np.random.default_rng(seed)
    coefs = rng.normal(**kwargs)  # dimensions should be state x actions
    # info = (seed, epsilon, coefs)
    remainder = epsilon / coefs.shape[1]

    def policy(state_samples: np.ndarray):
        """Policy function.

        :param state: should be dimension samples x state
        :return: array of samples x actions where each row is probability of
        each action.
        """
        return (1 - epsilon) * scipy.special.softmax(state_samples.dot(coefs),
                                                     axis=1) + remainder

    return policy


@pvalue('bounded_wor')
def bounded_wor(x,
                bounds: Tuple[float, float],
                N: int,
                null_mean: float,
                alpha: float,
                style: str,
                non_null_mean=3,
                version='hoef',
                gamma_series='easy',
                gamma_exp=None):
    """Bounded WoR e-value."""
    # x.shape == (trials, hypotheses, max_seq_len)
    trials, hypotheses, n = x.shape
    big_N = hypotheses * n
    assert big_N == N
    i_s = np.arange(1, n + 1)
    i_tile = np.tile(i_s, (trials, hypotheses, 1))
    bound_range = bounds[1] - bounds[0]
    if version == 'hoef':
        if style != 'invert':
            lam = np.sqrt(8 * np.log(hypotheses / (2 * alpha)) /
                          (n * np.square(bound_range)))

            mean_diffs = np.concatenate([
                np.zeros(shape=(trials, hypotheses, 1)),
                np.cumsum(x - null_mean, axis=2)[:, :, :-1]
            ],
                                        axis=2)
            assert i_tile.shape == mean_diffs.shape
            # print(mean_diffs[0, 0, :])
            outcomes = x - null_mean + (mean_diffs / (N - i_tile + 1))
            psi = np.square(lam * bound_range) / 8
            deltas = lam * outcomes - psi
            log_e_t = np.cumsum(deltas, axis=2)
            log_final_e = log_e_t[:, :, -1]

            log_max_e = np.max(log_e_t, axis=2)
            if style == 'stopped':
                raw_gammas = make_raw_gamma_coefs(np.arange(1, hypotheses + 1),
                                                  series=gamma_series,
                                                  exponent=gamma_exp)
                norm_gammas = raw_gammas / np.sum(raw_gammas)
                threshold = 1 / (norm_gammas * alpha)
                stop_level = np.tile(np.log(threshold), (trials, 1))
                stop_cond = log_max_e >= stop_level

                stop_idxs = np.where(stop_cond)
                sample_stop_idxs = np.argmax(
                    (log_e_t >= stop_level[:, :, np.newaxis])[stop_idxs],
                    axis=1)
                log_stop_e = np.zeros((trials, hypotheses))
                log_stop_e[stop_idxs] = log_e_t[stop_idxs[0], stop_idxs[1],
                                                sample_stop_idxs]
                log_stop_e[np.where(~stop_cond)] = log_final_e[np.where(
                    ~stop_cond)]

                # log_stop_e = stop_cond.astype(float) * log_max_e + (
                #    ~stop_cond).astype(float) * log_final_e
                # don't take max for ones that go into e-value algs b/c of
                # randomization techniques
                p_vals = np.exp(-1 * log_stop_e)
            elif style == 'sup':
                p_vals = np.exp(-1 * np.maximum(log_max_e, 0))
            else:
                assert style == 'final'
                p_vals = np.exp(-1 * np.maximum(log_final_e, 0))

            return p_vals
        else:
            sub_sums = np.cumsum(x, axis=2)
            denoms = n + np.sum((i_s - 1) / (N - i_s + 1))
            mu_hats = np.sum(x + sub_sums / (N - i_tile + 1), axis=2) / denoms

            def boundary_fn(mu_hat):
                num = bound_range / np.sqrt(2)
                denom = np.sqrt(n) + (np.sum(
                    (i_s - 1) / (N - i_s + 1)) / np.sqrt(n))
                fac = num / denom

                def search_fn(alpha):
                    if alpha <= 0:
                        return -np.inf
                    return mu_hat - null_mean - (fac *
                                                 np.sqrt(np.log(1 / alpha)))

                return search_fn

            result = np.ones((x.shape[0], x.shape[1]))
            for i, j in product(range(trials), range(hypotheses)):
                if mu_hats[i, j] <= null_mean:
                    result[i, j] = 1.
                else:
                    result[i, j] = scipy.optimize.root_scalar(boundary_fn(
                        mu_hats[i, j]),
                                                              bracket=(0, 1),
                                                              x0=0.05,
                                                              xtol=1e-8).root
            return result
    else:
        assert version == 'bet'

        def e_fn(x):
            m = (null_mean - bounds[0]) / (bound_range)

            def lam_fn(x, m):
                return lambda_aKelly(x, m, N=big_N)

            mart = betting_mart((x - bounds[0]) / bound_range,
                                m,
                                lambdas_fn_positive=lam_fn,
                                lambdas_fn_negative=lam_fn,
                                N=big_N)
            return np.max(mart) if stopped else mart[-1]

        e_vals = np.apply_along_axis(func1d=e_fn, axis=2, arr=x)
        return 1 / np.maximum(e_vals, 1)


@pvalue('e_contextual_bandit')
def e_contextual_bandit(x, policy_key, policy_kwargs, seeds, threshold):
    # Get data
    dataset = obp.dataset.OpenBanditDataset(behavior_policy='random',
                                            campaign='all')
    bandit_feedback = dataset.obtain_batch_bandit_feedback()
    policy_maker = get_policy(policy_key)
    for seed in seeds:
        policy = policy_maker(seed, **policy_kwargs)
        action_probs = policy(bandit_feedback['context'])
        sel_idxs = np.array(
            [np.array(x) for x in enumerate(bandit_feedback['action'])])
        ipw = action_probs[sel_idxs] / bandit_feedback['pscore']
        rewards = ipw * bandit_feedback['reward']
