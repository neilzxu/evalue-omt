"""Experiment that compares the the duration/distance of local dependence (lag)
with the power of each method.

The local dependence lag really only affects LORD^*, since e-LOND and
LOND already assume arbitrary dependence among all hypotheses already.
"""
from typing import Any, List, Tuple

from itertools import product
import os
import pickle
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

import numpy as np  # NOQA
import pandas as pd
import seaborn as sns

import metrics
from plot import plot_twinx_grid, add_title_legend, make_aligned_ticks
from utils import save_fig
from ..datasets import signal_comp_localdep_dataset
from ..utils import register_experiment, run_exp


def make_datapoints(results):
    def get_dps(result):
        fdps = metrics.fdp(result['alternates'], result['rejsets'])[:, -1]
        tdps = metrics.tdp(result['alternates'], result['rejsets'])[:, -1]
        return fdps, tdps, [alg.alpha for alg in result['instances']]

    datapoints = [{
        'Method':
        result['name'],
        '$\\pi_1$':
        result['data_spec']['kwargs']['non_null_p'],
        '$\\mu_1$':
        result['data_spec']['kwargs']['non_null_mean'],
        '$L$':
        result['data_spec']['kwargs']['lag'],
        'Neg':
        result['data_spec']['kwargs']['neg']
        if 'neg' in result['data_spec']['kwargs'] else False,
        'FDR':
        fdp,
        'Power':
        tdp,
        'alpha_levels':
        alpha
    } for result in results for fdp, tdp, alpha in zip(*get_dps(result))]
    return datapoints


def debug_data(df, out_dir):
    mask = (df['$L$'] == 400) & (df['$\\mu_1$'] == 1) & (df['Neg'])
    filt_df = df[mask]
    fig, ax = plt.subplots()
    for method in filt_df['Method'].unique():
        sub_df = filt_df[filt_df['Method'] == method]
        alpha_means = np.array(sub_df['alpha_levels'].tolist()).mean(axis=0)
        ax.plot(np.arange(alpha_means.shape[0]) + 1, alpha_means, label=method)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/alphas.png', dpi=300)
    plt.close(fig)


def setup_df(results: List[Any]):
    datapoints = make_datapoints(results)
    df = pd.DataFrame.from_records(datapoints)
    return df


alg_names = ['e-LOND', 'U-LOND', 'r-LOND', 'Ur-LOND', 'LORD$^*$']
# alg_names = ['e-LOND', 'U-LOND', 'r-LOND', 'Ur-LOND']
# alg_names = ['e-LOND', 'r-LOND']


def build_alg_specs(delta: float, hypotheses: int, lag: int,
                    bounds: Tuple[float, float]):
    gamma_dict = {'gamma_series': 'easy', 'gamma_exp': None}
    var = np.square(bounds[1] - bounds[0]) * 2 / 8
    p_spec = {
        'method': 'e_gaussian_mgf',
        'kwargs': {
            'alpha': delta,
            'version': 'inf',
            'var': var,
            **gamma_dict
        }
    }
    e_spec = {
        'method': 'e_gaussian_mgf',
        'kwargs': {
            'alpha': delta,
            'var': var,
            'version': 'mart_stopped',
            **gamma_dict
        }
    }
    ELOND_spec = {
        'method': 'LOND',
        'kwargs': {
            'delta': delta,
            'reshaping': None,
            'hypotheses': hypotheses,
            **gamma_dict
        },
        'pvalue': e_spec
    }
    rLOND_spec = {
        'method': 'LOND',
        'kwargs': {
            'delta': delta,
            'reshaping': 'BY',
            'hypotheses': hypotheses,
            **gamma_dict
        },
        'pvalue': p_spec
    }
    ULOND_spec = {
        'method': 'LOND',
        'kwargs': {
            'delta': delta,
            'reshaping': None,
            'rand': 'ind',
            'hypotheses': hypotheses,
            **gamma_dict
        },
        'pvalue': e_spec
    }
    UrLOND_spec = {
        'method': 'LOND',
        'kwargs': {
            'delta': delta,
            'reshaping': 'BY',
            'rand': 'ind',
            'hypotheses': hypotheses,
            **gamma_dict
        },
        'pvalue': p_spec
    }
    LORD_star_spec = {
        'method': 'LORD',
        'kwargs': {
            'delta': delta,
            'bound': None,
            'startfac': 0.9,
            'alpha_strategy': 2,
            'lag': lag,
            'hypotheses': hypotheses,
            **gamma_dict
        },
        'pvalue': p_spec
    }
    return [ELOND_spec, ULOND_spec, rLOND_spec, UrLOND_spec, LORD_star_spec]
    # return [ELOND_spec, ULOND_spec, rLOND_spec, UrLOND_spec]


@register_experiment('lag_comp')
def lag_comp(*, processes: int, out_dir: str, result_dir: str,
             save_result: bool, custom_param: str, **kwargs) -> None:

    trials = 200
    hypotheses = int(1e3)
    rho = 0.5
    delta = 0.3
    max_seq_len = 200
    signals = [2.5, 3]

    probs = [0.3]
    ab_sum = 0.01
    bounds = (-4, 4)
    assert np.max(signals) < bounds[1]
    assert np.min(signals) > bounds[0]

    # Generate specs
    lags = list(range(0, 550, 50))
    # custom_lag = lags[int(custom_param)]
    # lags = [0]
    signal_comps = [
        signal_comp_localdep_dataset(mode='bounded',
                                     ab_sum=ab_sum,
                                     bounds=bounds,
                                     signals=signals,
                                     probs=probs,
                                     lag=lag,
                                     rho=rho,
                                     max_seq_len=max_seq_len,
                                     hypotheses=hypotheses,
                                     neg=neg,
                                     trials=trials)
        for lag, neg in product(lags, [True])  # if lag == custom_lag
    ]

    data_specs = [
        data_spec for data_comp in signal_comps for data_spec in data_comp
    ]
    all_names = alg_names * len(data_specs)
    all_flat_args = []
    for data_idx, data_spec in enumerate(data_specs):
        lag = data_spec['kwargs']['lag']

        alg_specs = build_alg_specs(delta=delta,
                                    hypotheses=hypotheses,
                                    lag=lag,
                                    bounds=bounds)
        assert len(alg_specs) == len(alg_names)

        all_flat_args += [(data_idx, alg_spec) for alg_spec in alg_specs]

    # all_df_paths = [f'{out_dir}/results_{idx}.csv' for idx in range(len(lags))]
    # result_df_path = f'{out_dir}/results_{custom_param}.csv'
    result_df_path = f'{out_dir}/results.csv'
    if not os.path.exists(result_df_path):
        results = run_exp(out_dir,
                          alg_specs=None,
                          data_specs=data_specs,
                          data_alg_names=all_names,
                          processes=processes,
                          data_alg_specs=all_flat_args,
                          save_result=save_result,
                          save_data=False,
                          lazy_data=True,
                          cache_data=True,
                          long_exp=False)
        if save_result:
            results = [
                pickle.load(open(os.path.join(out_dir, filename), 'rb'))
                for filename in os.listdir(out_dir)
                if re.fullmatch(r'.*\.pkl', filename)
            ]
        df = setup_df(results)
        df.to_csv(result_df_path)
    else:
        df = pd.read_csv(result_df_path)
    # debug_data(df, out_dir)
    df = df.drop('alpha_levels', axis=1)

    # if custom_param == len(lags):
    # df = pd.concat([pd.read_csv(df_path) for df_path in all_df_paths])
    sns.set_theme()
    for neg, prob in product([True], probs):
        mask = (df['Neg'] == neg) & (df['$\\pi_1$']
                                     == prob) & (df['Method'] != 'U-LOND')
        neg_df = df[mask]
        line_kws = {
            'y_kws': {
                'linestyle': 'solid',
                'linewidth': 2,
                'markersize': 3
            },
            'twin_kws': {
                'linestyle': 'dashed',
                'linewidth': 2,
                'markersize': 3
            }
        }
        hue_kws = {
            'marker': ["o", "X", "D", "P", "s"],
            'color': sns.set_palette("tab10")
        }

        sns.set_theme()
        res = plot_twinx_grid(
            neg_df,
            x='$L$',
            y='Power',
            twin_y='FDR',
            hue='Method',
            row=None,
            col='$\\mu_1$',
            hue_kws=hue_kws,
            width=10,
            height=4,
            hue_order=[name for name in alg_names if name != 'U-LOND'],
            error_bar_kwargs={},
            **line_kws)
        fg, plot_map, hue_lh, twin_lh = res[0], res[1], res[-2], res[-1]

        # Share x axes (y and twin_y axes set explicitly later)
        list(plot_map.values())[0][0].get_shared_x_axes().join(
            *[value[0] for value in plot_map.values()])

        # Create ticks that are aligned between plots (Power, FDR)

        # Format each plot appropriately
        for signal, (ax_power, ax_fdr) in plot_map.items():
            if signal == 3:
                power_ylim, fdr_ylim = (0, 0.8), (0, 0.2)
            else:
                power_ylim, fdr_ylim = (0, 0.3), (0, 0.2)
            lim_tick_pairs = make_aligned_ticks(ranges=[power_ylim, fdr_ylim],
                                                n_ticks=5,
                                                start_pad=0.3,
                                                end_pad=0.3)
            for ax, (lim, ticks), ylabel in zip([ax_power, ax_fdr],
                                                lim_tick_pairs,
                                                ['Power', 'FDR']):
                ax.set_ylim(lim)
                ax.set_yticks(ticks)
                ax.tick_params(axis='y', length=0)
                ax.set_ylabel(ylabel)
                ax.set_title(f'$\\mu_1={signal}$')
                # ax.set_yscale('log')
            ax_fdr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax_power.set_xlabel('$L$')

        # Organize plot size and add legends for method and metric
        fg.tight_layout(rect=[0, 0.1, 1, 1], pad=1)

        class TitleLegendHandler(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize, trans):
                # Create a small empty rectangle for the title
                title_patch = mpatches.Rectangle((0, 0),
                                                 0,
                                                 0,
                                                 fill=False,
                                                 edgecolor='none',
                                                 linewidth=0)
                title_text = plt.Text(0,
                                      0,
                                      orig_handle.get_label(),
                                      fontsize=fontsize)

                return [title_patch, title_text]

        fg.legend(*(add_title_legend(*hue_lh, 'Method')),
                  handler_map={mpatches.Patch: TitleLegendHandler()},
                  loc='center left',
                  ncol=len(hue_lh[0]) + 1,
                  bbox_to_anchor=(0, 0.005, 0.5, 0.1),
                  columnspacing=.8)
        fg.legend(*(add_title_legend(*twin_lh, 'Metric')),
                  handler_map={mpatches.Patch: TitleLegendHandler()},
                  loc='center',
                  ncol=len(twin_lh[0]) + 1,
                  bbox_to_anchor=(0.5, 0.005, 0.5, 0.1),
                  columnspacing=.8)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        save_fig(fg.figure, result_dir,
                 f'all_metrics_neg={neg}_prob={prob:.1f}')
