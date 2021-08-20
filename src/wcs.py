"""Code borrowed heavily and adapted from https://github.com/ying531/conformal-
selection."""
import numpy as np
import pandas as pd

from alg.LOND import LOND

_evalue_dispatch = {}


def evalue(name):
    def register(fn):
        _evalue_dispatch[name] = fn
        return fn

    return register


def _weighted_ct(s_calib_scores, s_calib_cum_weights, test_score):
    left_idx = np.searchsorted(s_calib_scores, test_score, side='left')
    right_idx = np.searchsorted(s_calib_scores, test_score, side='right')
    if right_idx == 0:
        weighted_less_sum = weighted_equal_sum = 0.
    elif left_idx == 0:
        weighted_less_sum = 0
        weighted_equal_sum = s_calib_cum_weights[right_idx - 1]
    else:
        weighted_less_sum = s_calib_cum_weights[left_idx - 1]
        weighted_equal_sum = s_calib_cum_weights[right_idx -
                                                 1] - weighted_less_sum
    return weighted_less_sum, weighted_equal_sum


def _make_up_low(s_calib_scores, s_calib_cum_weights, test_score, test_weight):
    less_sum, equal_sum = _weighted_ct(s_calib_scores, s_calib_cum_weights,
                                       test_score)
    up = less_sum + equal_sum + test_weight
    low = less_sum + equal_sum
    return low, up


def _make_sum_val(s_calib_scores, s_calib_cum_weights, test_score, test_weight,
                  u):
    less_sum, equal_sum = _weighted_ct(s_calib_scores, s_calib_cum_weights,
                                       test_score)
    res = less_sum + (u * (equal_sum + test_weight))
    return res


def weighted_pvals(calib_scores,
                   calib_weights,
                   test_scores,
                   test_weights,
                   s_calib_scores,
                   s_calib_cum_weights,
                   seed=None):
    rng = np.random.default_rng(seed)
    sum_calib_weight = np.sum(calib_weights)

    # s_calib_idxs = np.argsort(calib_scores)
    # s_calib_scores = calib_scores[s_calib_idxs]
    # s_calib_cum_weights = np.cumsum(calib_weights[s_calib_idxs])

    T = len(test_scores)
    pvals = np.zeros(T)
    for j in range(T):
        u = rng.uniform(size=1)[0]
        # pval = (np.sum(calib_weights[calib_scores < test_scores[j]]) +
        #         (np.sum(calib_weights[calib_scores == test_scores[j]]) +
        #          test_weights[j]) * u) / (sum_calib_weight + test_weights[j])
        pval = _make_sum_val(s_calib_scores, s_calib_cum_weights,
                             test_scores[j], test_weights[j],
                             u) / (sum_calib_weight + test_weights[j])
        # assert np.isclose(dummy_pval, pval), (dummy_pval, pval)
        pvals[j] = pval
    # pvals = pvals / (sum_calib_weight + test_weights)
    # print(pvals)
    return pvals


@evalue('lond')
def lond_e(calib_scores,
           calib_weights,
           test_scores,
           test_weights,
           s_calib_scores,
           s_calib_cum_weights,
           alpha,
           lond_kwargs,
           gamma_factor=1,
           seed=None):
    rng = np.random.default_rng(seed)
    sum_calib_weight = np.sum(calib_weights)
    # s_calib_idxs = np.argsort(calib_scores)
    # s_calib_scores = calib_scores[s_calib_idxs]
    # s_calib_cum_weights = np.cumsum(calib_weights[s_calib_idxs])
    T = len(test_scores)
    omts = []
    inv_e_values = []
    for j in range(T):
        # compute all other pvals
        if j >= 1:
            pval_j_low = np.zeros(j - 1)
            pval_j_up = np.zeros(j - 1)
            for k in range(j - 1):
                # pval_j_up[k] = np.sum(calib_weights[
                #     calib_scores < test_scores[k]]) + test_weights[j]
                # pval_j_low[k] = np.sum(calib_weights[
                #     calib_scores < test_scores[k]])  # lower bound on value
                pval_j_low[k], pval_j_up[k] = _make_up_low(
                    s_calib_scores, s_calib_cum_weights, test_scores[k],
                    test_weights[j])
                # assert np.isclose(dummy_low,
                #                   pval_j_low[k]), (dummy_low, pval_j_low[k])
                # assert np.isclose(dummy_up,
                #                   pval_j_up[k]), (dummy_low, pval_j_up[k])
            pval_j_up = pval_j_up / (sum_calib_weight + test_weights[j])
            pval_j_low = pval_j_low / (sum_calib_weight + test_weights[j])
        u = rng.uniform(size=1)[0]
        # pval = (np.sum(calib_weights[calib_scores < test_scores[j]]) +
        #         (np.sum(calib_weights[calib_scores == test_scores[j]]) +
        #          test_weights[j]) * u) / (sum_calib_weight + test_weights[j])
        pval = _make_sum_val(s_calib_scores, s_calib_cum_weights,
                             test_scores[j], test_weights[j],
                             u) / (sum_calib_weight + test_weights[j], )
        # assert np.isclose(dummy_pval, pval), (dummy_pval, pval)
        omt_low = LOND(delta=alpha,
                       reshaping=None,
                       hypotheses=T,
                       rand=None,
                       **lond_kwargs)
        if j >= 1:
            for p in pval_j_low:
                omt_low.process_next(p)

        omt_up = LOND(delta=alpha,
                      reshaping=None,
                      hypotheses=T,
                      rand=None,
                      **lond_kwargs)
        if j >= 1:
            for p in pval_j_up:
                omt_up.process_next(p)
        low_test_level = omt_low._next_alpha()
        up_test_level = omt_up._next_alpha()
        inv_e_value = low_test_level if pval <= up_test_level else np.inf

        inv_e_values.append(inv_e_value)
        omts.append((omt_low, omt_up))
    # print(inv_e_values)
    return inv_e_values, omts


@evalue('wcs')
def wcs_e(calib_scores,
          calib_weights,
          test_scores,
          test_weights,
          alpha,
          q,
          seed=None):
    sum_calib_weight = np.sum(calib_weights)

    rng = np.random.default_rng(seed)
    ntest = len(test_scores)
    inv_e_values = []

    for j in range(ntest):
        # compute all other pvals
        j_test = j + 2
        pval_j = np.zeros(j + 1)
        for k in range(j + 1):
            if k != j:
                pval_j[k] = np.sum(
                    calib_weights[calib_scores < test_scores[k]]
                ) + test_weights[j] * (test_scores[j] < test_scores[k])
                # might need bug fix --- original code used k instead of j to
                # index test_weight. and the papers uses j, but i think k works
                # too lol
        pval_j = pval_j / (sum_calib_weight + test_weights[j])
        pval = (np.sum(calib_weights[calib_scores < test_scores[j]]) +
                (np.sum(calib_weights[calib_scores == test_scores[j]]) +
                 test_weights[j]) * rng.uniform(size=1)[0]) / (
                     sum_calib_weight + test_weights[j])

        # run BH
        df_j = pd.DataFrame({
            "id": range(j_test),
            "pval": np.append(pval_j, 0)
        }).sort_values(by='pval')
        df_j['threshold'] = q * np.linspace(1, j_test, num=j_test) / (j_test)
        idx_small_j = [
            s for s in range(j_test) if df_j.iloc[s, 1] <= df_j.iloc[s, 2]
        ]
        Rj = np.array(df_j['id'])[range(np.max(idx_small_j) + 1)]
        test_level = q * Rj[-1] / (j_test)
        inv_e_value = test_level if pval <= test_level else np.inf
        inv_e_values.append(inv_e_value)
    return inv_e_values, None


# _REJSET_DEBUG = []

# def rejset_release(out_dir):
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)


def elond(calib_scores,
          calib_weights,
          test_scores,
          test_weights,
          s_calib_scores,
          s_calib_cum_weights,
          alpha: float,
          rand=None,
          evalue: str = 'lond',
          seed=None) -> np.ndarray:
    evalue_kwargs = {
        'lond_kwargs': {
            'gamma_series': 'easy'
        }
    } if evalue == 'lond' else {
        'q': alpha
    }
    # wpvals = weighted_pvals(calib_scores, calib_weights, test_scores,
    #                         test_weights)
    inv_e_t, evalue_omt = _evalue_dispatch[evalue](calib_scores,
                                                   calib_weights,
                                                   test_scores,
                                                   test_weights,
                                                   s_calib_scores,
                                                   s_calib_cum_weights,
                                                   alpha,
                                                   seed=seed + 1,
                                                   **evalue_kwargs)
    elond_kwargs = {'gamma_series': 'easy', 'seed': seed}
    omt = LOND(delta=alpha,
               reshaping=None,
               rand=rand,
               hypotheses=len(test_scores),
               **elond_kwargs)
    omt.run_fdr(inv_e_t)

    # not_inf = np.array(inv_e_t) < np.inf
    # if rand is None and np.any(not_inf):
    #     print('evalue', evalue)
    #     print(f'not inf count: {np.sum(not_inf)}')
    #     print(np.where(not_inf))
    #     #print('inv_e_t', inv_e_t)
    #     print('weighted pval', wpvals[not_inf])
    #     print('inv e value test level', np.array(inv_e_t)[not_inf])
    #     print('real test value', np.array(omt.alpha)[not_inf])
    #     print(omt.rej_indices)
    # r_t = np.zeros(len(test_scores))
    # r_t[omt.rej_indices] = 1
    # r_t = np.cumsum(r_t)

    #_REJSET_DEBUG.append(pd.DataFrame{'t': np.arange(len())})
    return np.array(omt.rej_indices)


def elond_wcs(calib_scores,
              calib_weights,
              test_scores,
              test_weights,
              alpha: float,
              seed=None) -> np.ndarray:
    return elond(calib_scores,
                 calib_weights,
                 test_scores,
                 test_weights,
                 alpha,
                 rand=None,
                 evalue='wcs',
                 seed=seed)


def ulond_wcs(calib_scores,
              calib_weights,
              test_scores,
              test_weights,
              alpha: float,
              seed=None) -> np.ndarray:
    return elond(calib_scores,
                 calib_weights,
                 test_scores,
                 test_weights,
                 alpha,
                 rand='ind',
                 evalue='wcs',
                 seed=seed)


def ulond(calib_scores,
          calib_weights,
          test_scores,
          test_weights,
          s_calib_scores,
          s_calib_cum_weights,
          alpha: float,
          rand=None,
          evalue: str = 'lond',
          seed=None) -> np.ndarray:
    return elond(calib_scores,
                 calib_weights,
                 test_scores,
                 test_weights,
                 s_calib_scores,
                 s_calib_cum_weights,
                 alpha,
                 rand='ind',
                 evalue=evalue,
                 seed=seed)


def rlond(calib_scores,
          calib_weights,
          test_scores,
          test_weights,
          s_calib_scores,
          s_calib_cum_weights,
          alpha: float,
          rand=None,
          seed=None) -> np.ndarray:
    pvals = weighted_pvals(calib_scores,
                           calib_weights,
                           test_scores,
                           test_weights,
                           s_calib_scores,
                           s_calib_cum_weights,
                           seed=seed + 1)
    rlond_kwargs = {'gamma_series': 'easy', 'seed': seed}
    omt = LOND(delta=alpha,
               reshaping='BY',
               rand=rand,
               hypotheses=len(test_scores),
               **rlond_kwargs)
    omt.run_fdr(pvals)
    return np.array(omt.rej_indices)


def urlond(calib_scores,
           calib_weights,
           test_scores,
           test_weights,
           s_calib_scores,
           s_calib_cum_weights,
           alpha: float,
           rand=None,
           seed=None) -> np.ndarray:
    return rlond(calib_scores,
                 calib_weights,
                 test_scores,
                 test_weights,
                 s_calib_scores,
                 s_calib_cum_weights,
                 alpha,
                 rand='ind',
                 seed=seed)


def lond(calib_scores,
         calib_weights,
         test_scores,
         test_weights,
         alpha: float,
         seed=None) -> np.ndarray:
    pvals = weighted_pvals(calib_scores, calib_weights, test_scores,
                           test_weights)
    lond_kwargs = {'gamma_series': 'easy', 'seed': seed}
    omt = LOND(delta=alpha,
               reshaping=None,
               hypotheses=len(test_scores),
               **lond_kwargs)
    omt.run_fdr(pvals)
    return np.array(omt.rej_indices)
