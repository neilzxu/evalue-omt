"""Experiment that compares LOND and e-LOND under bounded sampling WoR."""

from itertools import product
import os

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
import numpy as np  # NOQA
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import metrics
from plot import plot_twinx_grid, add_title_legend, make_aligned_ticks
from ..utils import register_experiment, run_exp

fixed_data_kwargs = {
    'null_mean': 0.,
    'ab_sum': 0.01,
    'bounds': (-4, 4),
    'sample_size': 100,
    'hypotheses': 1000,
    'trials': 500
}
delta = 0.05


def debug_datapoints(results, result_dir):
    quantiles = [0.25, 0.5, 0.75]
    result = results[0]
    items = [('pvalue', result['pvalues']),
             ('alphas',
              np.array([instance.alpha for instance in result['instances']])),
             ('fdp', metrics.fdp(result['alternates'], result['rejsets']))]
    for name, values in tqdm(items, desc="Plotting Exp5 plots"):
        fig, ax = plt.figure(), plt.gca()
        for quantile in quantiles:
            p_quant = np.quantile(values, quantile, axis=0)
            ax.plot(np.arange(values.shape[1]),
                    p_quant,
                    label=f'{result["name"]} quantile={quantile:.2f}')
        p_quant = np.mean(values, axis=0)
        ax.plot(np.arange(values.shape[1]),
                p_quant,
                label=f'{result["name"]} mean')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{result_dir}/{name}_quantiles.png', dpi=300)
        plt.close(fig)
        if name == 'pvalue':
            fig, ax = plt.figure(), plt.gca()
            ax.hist(values.flatten(), bins=100)
            ax.set_xscale('log')
            fig.tight_layout()
            fig.savefig(f'{result_dir}/pvalue_hist.png', dpi=300)
            plt.close(fig)


def make_datapoints(results):
    def get_dps(result):
        fdps = metrics.fdp(result['alternates'], result['rejsets'])[:, -1]
        tdps = metrics.tdp(result['alternates'], result['rejsets'])[:, -1]
        return fdps, tdps

    datapoints = [{
        'Method': result['name'],
        '$\\pi_1$': result['data_spec']['kwargs']['non_null_p'],
        '$\\mu_1$': result['data_spec']['kwargs']['non_null_mean'],
        'FDR': fdp,
        'Power': tdp
    } for result in results for fdp, tdp in zip(*get_dps(result))]
    return datapoints


def setup_df(results, out_dir: str):
    datapoints = make_datapoints(results)
    df = pd.DataFrame.from_records(datapoints)
    return df


def make_data_specs():
    seed_start = 322
    p_range = np.arange(0.0, 0.5, 0.05)
    # p_range = [0.0]
    mu_range = np.array([1.5, 2.])
    data_specs = [{
        'method': 'discrete_bounded_wor',
        'kwargs': {
            'non_null_p': non_null_p,
            'non_null_mean': non_null_mean,
            'seed': seed_start + idx,
            **fixed_data_kwargs
        }
    } for idx, (non_null_p,
                non_null_mean) in enumerate(product(p_range, mu_range))]
    return data_specs


def make_alg_specs():
    N = fixed_data_kwargs['hypotheses'] * fixed_data_kwargs['sample_size']
    gamma_dict = {
        'delta': delta,
        'gamma_series': 'easy',
        'gamma_exp': None,
        'hypotheses': fixed_data_kwargs['hypotheses']
    }

    e_spec = {
        'method': 'bounded_wor',
        'kwargs': {
            'bounds': fixed_data_kwargs['bounds'],
            'N': N,
            'null_mean': fixed_data_kwargs['null_mean'],
            'alpha': delta,
            'style': 'stopped',
            'version': 'hoef'
        }
    }
    p_spec = {
        'method': 'bounded_wor',
        'kwargs': {
            'bounds': fixed_data_kwargs['bounds'],
            'N': N,
            'null_mean': fixed_data_kwargs['null_mean'],
            'alpha': delta,
            'style': 'sup',
            'version': 'hoef'
        }
    }

    ELOND_alg = {
        'method': 'LOND',
        'kwargs': {
            'reshaping': None,
            **gamma_dict
        },
        'pvalue': e_spec
    }
    ULOND_alg = {
        'method': 'LOND',
        'kwargs': {
            'reshaping': None,
            'rand': 'ind',
            **gamma_dict
        },
        'pvalue': e_spec
    }
    LOND_alg = {
        'method': 'LOND',
        'kwargs': {
            'reshaping': 'BY',
            **gamma_dict
        },
        'pvalue': p_spec
    }
    UrLOND_alg = {
        'method': 'LOND',
        'kwargs': {
            'reshaping': 'BY',
            'rand': 'ind',
            **gamma_dict
        },
        'pvalue': p_spec
    }
    return (['e-LOND', 'r-LOND', 'Ur-LOND',
             'U-LOND'], [ELOND_alg, LOND_alg, UrLOND_alg, ULOND_alg])


@register_experiment('wor_comp')
def wor_comp(processes: int, out_dir: str, result_dir: str, save_result: bool,
             **kwargs) -> None:
    # Load and run experiments
    data_specs = make_data_specs()
    alg_names, alg_specs = make_alg_specs()
    def_cmap = plt.get_cmap('Set1')
    df_path = f'{out_dir}/results.csv'
    if not os.path.exists(df_path):
        results = run_exp(out_dir,
                          alg_specs=alg_specs,
                          data_specs=data_specs,
                          alg_names=alg_names,
                          processes=processes,
                          save_result=save_result)
        df = setup_df(results, out_dir)
        print(df)
        df.to_csv(df_path)
    else:
        # Load results
        df = pd.read_csv(df_path)

    def err_fn(x):
        width = x.std() / np.sqrt(len(x)) * 1.96
        mean = x.mean()
        return mean - width, mean + width

    # Plot results
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
        'marker': ["o", "X", "D", "s"],
        'color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    }

    sns.set_theme()
    res = plot_twinx_grid(df,
                          x='$\\pi_1$',
                          y='Power',
                          twin_y='FDR',
                          hue='Method',
                          row=None,
                          col='$\\mu_1$',
                          hue_kws=hue_kws,
                          width=10,
                          height=4,
                          **line_kws)
    fg, plot_map, hue_lh, twin_lh = res[0], res[1], res[-2], res[-1]
    # Share x axes (y and twin_y axes set explicitly later)
    list(plot_map.values())[0][0].get_shared_x_axes().join(
        *[value[0] for value in plot_map.values()])

    # Create ticks that are aligned between plots (Power, FDR)

    # Format each plot appropriately
    for signal, (ax_power, ax_fdr) in plot_map.items():
        if signal == 2:
            power_ylim, fdr_ylim = (0, 1.), (0, 0.3)
        else:
            power_ylim, fdr_ylim = (0, 1.), (0, 0.3)
        lim_tick_pairs = make_aligned_ticks(ranges=[power_ylim, fdr_ylim],
                                            n_ticks=5,
                                            start_pad=0.3,
                                            end_pad=0.3)
        for ax, (lim, ticks), ylabel in zip([ax_power, ax_fdr], lim_tick_pairs,
                                            ['Power', 'FDR']):
            ax.set_ylim(lim)
            ax.set_yticks(ticks)
            ax.tick_params(axis='y', length=0)
            ax.set_ylabel(ylabel)
            ax.set_title(f'$\\mu_1={signal}$')
            # ax.set_yscale('log')
        ax_fdr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_power.set_xlabel('$\\pi_1$')
        ax.axhline(delta, color='gray', linestyle='dashed')

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

    # Organize plot size and add legends for method and metric
    fg.tight_layout(rect=[0, 0.1, 1, 1], pad=1)
    fg.legend(*(add_title_legend(*hue_lh, 'Method')),
              handler_map={mpatches.Patch: TitleLegendHandler()},
              loc='center',
              ncol=len(hue_lh[0]) + 1,
              bbox_to_anchor=(0, 0.005, 0.6, 0.1),
              columnspacing=.8)
    fg.legend(*(add_title_legend(*twin_lh, 'Metric')),
              loc='center',
              handler_map={mpatches.Patch: TitleLegendHandler()},
              ncol=len(twin_lh[0]) + 1,
              bbox_to_anchor=(0.5, 0.005, 0.5, 0.1),
              columnspacing=.8)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fg.savefig(f'{result_dir}/all_metrics.pdf')
    # debug_datapoints(results, result_dir)
