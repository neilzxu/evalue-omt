from typing import Any, Dict, List, Tuple, Optional
from itertools import product

import numpy as np

DataSpec = Tuple[str, Dict[str, Any]]


def hmm_dataset(non_null_mean_min: int,
                non_null_mean_max: int,
                trans_min: float = 0.1,
                trans_max: float = 1,
                trans_step: float = 0.1,
                init_p: float = 0.5) -> List[DataSpec]:
    """Produces spec associated with HMM datasets.

    Range of signals in datasets span integer values [non_null_mean_min,
    non_null_mean_max] and transition probabilities at trans_step
    intervals from trans_min to trans_max (exclusive). init_p specifies
    the parameter for the Bernoulli RV determining whether the initial
    hypothesis is non-null.
    """
    param_list = list(
        product(np.arange(trans_min, trans_max, trans_step),
                range(non_null_mean_min, non_null_mean_max + 1)))
    seeds = range(322, 322 + len(param_list))
    return [{
        'method': 'hmm',
        'kwargs': {
            'trials': 200,
            'hypotheses': 1000,
            'transition_prob': transition_prob,
            'non_null_mean': non_null_mean,
            'init_p': init_p,
            'seed': seed
        }
    } for seed, (transition_prob, non_null_mean) in zip(seeds, param_list)]


def signal_comp_dataset(signals: List[float],
                        probs: List[float],
                        hypotheses: int = 1000,
                        trials: int = 200) -> List[DataSpec]:
    """Produces spec associated with constant datasets.

    Range of signals in datasets span integer values in [signal_min,
    signal_max] with probabilities of non-null ranging from prob_min to
    prob_max (exclusive) in steps of prob_step.
    """
    gaussian_params = list(
        product(np.arange(prob_min, prob_max, prob_step),
                range(signal_min, signal_max + 1)))
    gaussian_seeds = range(322, 322 + len(gaussian_params))
    return [{
        'method': 'gaussian',
        "kwargs": {
            'trials': trials,
            'hypotheses': hypotheses,
            'non_null_mean': mean,
            'non_null_sd': 1,
            'null_mean': 0,
            'null_sd': 1,
            'non_null_p': p,
            'seed': seed
        }
    } for seed, (p, mean) in zip(gaussian_seeds, gaussian_params)]


def signal_comp_localdep_dataset(signals: List[float],
                                 probs: List[float],
                                 total_var: Optional[float] = None,
                                 mode: str = 'gaussian',
                                 max_seq_len: int = 50,
                                 neg: bool = False,
                                 lag: int = 100,
                                 rho: float = 0.5,
                                 hypotheses: int = 1000,
                                 trials: int = 200,
                                 bounds: Optional[Tuple[float, float]] = None,
                                 ab_sum: Optional[Tuple[float, float]] = None,
                                 seed: float = 322) -> List[DataSpec]:
    """Produces spec for local dependence data associated with constant
    datasets."""

    assert mode == 'bounded' or total_var is not None
    sample_signals = signals
    sample_var = total_var if total_var is not None else 1
    params = list(product(probs, sample_signals))
    seeds = range(seed, seed + len(params))

    bounded_kwargs = {'bounds': bounds, 'ab_sum': ab_sum}
    gaussian_kwargs = {'sd': np.sqrt(sample_var)}

    extra_kwargs, method = (
        bounded_kwargs, 'bounded') if mode == 'bounded' else (gaussian_kwargs,
                                                              'gaussian')
    return [{
        'method': f'localdep_{method}',
        'kwargs': {
            'trials': trials,
            'hypotheses': hypotheses,
            'non_null_mean': mean,
            'null_mean': 0,
            'non_null_p': p,
            'seed': seed,
            'lag': lag,
            'max_seq_len': max_seq_len,
            'rho': rho,
            'neg': neg,
            **extra_kwargs
        }
    } for seed, (p, mean) in zip(seeds, params)]
