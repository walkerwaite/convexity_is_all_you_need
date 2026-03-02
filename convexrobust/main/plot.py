from __future__ import annotations

import warnings
# Surpress lightning-bolt warnings https://github.com/Lightning-Universe/lightning-bolts/issues/563
warnings.simplefilter('ignore')
original_filterwarnings = warnings.filterwarnings
def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append':True})
warnings.filterwarnings = _filterwarnings

import torch

import numpy as np
import collections
import click

import os.path as op
import matplotlib

matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt

from convexrobust.data import datamodules
from convexrobust.model.certificate import Norm, Certificate
from convexrobust.utils import dirs, file_utils, pretty
from convexrobust.utils import torch_utils as TU
from convexrobust.main import main

from convexrobust.main.evaluate import Result, ResultDict

from sklearn.metrics import ConfusionMatrixDisplay

from collections import OrderedDict

from typing import Optional

def _abcrown_levels(global_params, norm, max_r):
    """Return the epsilon levels to show for ab-CROWN (pruned to plot range).
    Defaults to your mnist_38 script values. Fall back to a small auto grid otherwise."""
    # Known levels from your bash script for MNIST-3/8:
    levels_by_norm = {
        # L1
        "L1":  [2, 4, 6, 8, 10, 12, 14],
        # L2
        "L2":  [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
        # LInf
        "LInf":[0.025, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.125],
    }

    key = getattr(norm, "name", str(norm))
    levels = levels_by_norm.get(key, None)

    # If we don't have a hand-written list (other dataset), make a small auto grid.
    if not levels:
        # 6 evenly spaced levels between 0 and max_r (exclusive of 0)
        if max_r <= 0:
            return []
        levels = np.linspace(max_r/12, max_r, num=6).tolist()

    # Clip to plotable range and sorted unique
    print(levels)

    levels = sorted({x for x in levels if 0 < x <= max_r})
    return levels

def figs_path(file_name: str, global_params) -> str:
    return op.join(f'./figs/{global_params.experiment_directory}', file_name)


def clean_confusion_plot(results: ResultDict, global_params) -> None:
    for (name, res) in results.items():
        fig = plt.figure(figsize=(8, 6), dpi=200)
        targets = TU.numpy(torch.cat([r.target for r in res]))
        preds = TU.numpy(torch.cat([r.pred for r in res]))

        ConfusionMatrixDisplay.from_predictions(targets, preds, cmap='plasma', normalize='true')

        fig.tight_layout()
        plt.savefig(figs_path(f'{name}_confusion.png', global_params))


labels_dict = {
    'ablation_reg': OrderedDict([
        ('convex_reg_0', r'$\lambda=0.0$'),
        ('convex_reg_1', r'$\lambda=0.0025$'),
        ('convex_reg_2', r'$\lambda=0.005$'),
        ('convex_reg_3', r'$\lambda=0.0075$'),
        ('convex_reg_4', r'$\lambda=0.01$'),
    ]),
    'ablation_feature_map': OrderedDict([
        ('convex_nofeaturemap', r'$\varphi=\textrm{Id}$'),
        ('convex_reg_0', r'Standard $\varphi$'),
    ]),
}

for i in range(1, 5):
    labels_dict[f'standard_{i}'] = OrderedDict([
        ('convex_noreg', 'Convex* (Not regularized)'),
        ('convex_reg', 'Convex*'),
        (f'randsmooth_gauss_{i}', 'RS Gaussian'),
        (f'randsmooth_laplace_{i}', 'RS Laplacian'),
        (f'randsmooth_uniform_{i}', 'RS Uniform'),
        (f'randsmooth_splitderandomized_{i}', 'Splitting'),
        ('cayley', 'Cayley'),
        ('abcrown', r'$\alpha,\beta$-CROWN'),
        ('linf', r'$\ell_{\infty}$ Nets'),
    ])
    labels_dict[f'standard_{i}_splitting_4'] = OrderedDict([
        ('convex_noreg', 'Convex* (Not regularized)'),
        ('convex_reg', 'Convex*'),
        (f'randsmooth_gauss_{i}', 'RS Gaussian'),
        (f'randsmooth_laplace_{i}', 'RS Laplacian'),
        (f'randsmooth_uniform_{i}', 'RS Uniform'),
        ('randsmooth_splitderandomized_4', 'Splitting'),
        ('abcrown', r'$\alpha,\beta$-CROWN'),
        ('cayley', 'Cayley'),
        ('linf', r'$\ell_{\infty}$ Nets'),
    ])
    labels_dict[f'standard_{i}_splitting_large'] = OrderedDict([
        ('convex_noreg', 'Convex* (Not regularized)'),
        ('convex_reg', 'Convex*'),
        (f'randsmooth_gauss_{i}', 'RS Gaussian'),
        (f'randsmooth_laplace_{i}', 'RS Laplacian'),
        (f'randsmooth_uniform_{i}', 'RS Uniform'),
        ('randsmooth_splitderandomized_large', 'Splitting'),
        ('abcrown', r'$\alpha,\beta$-CROWN'),
        ('cayley', 'Cayley'),
        ('linf', r'$\ell_{\infty}$ Nets'),
    ])


# Used to reproduce the plots for the paper. From the hyperparam table in Section E of appendix
labels_dict['mnist_38_paper'] = labels_dict['standard_1_splitting_4']
labels_dict['fashion_mnist_shirts_paper'] = labels_dict['standard_1']
labels_dict['cifar10_catsdogs_paper'] = labels_dict['standard_2']
labels_dict['malimg_paper'] = labels_dict['standard_4_splitting_large']


norm_str_dict = {Norm.L1: r'$\ell_1$', Norm.L2: r'$\ell_2$', Norm.LInf: r'$\ell_{\infty}$'}

line_colors = ['#88CCEE', '#CC6677', '#DDCC77', '#117733',
               '#332288', '#AA4499', '#44AA99', '#999933',
               '#882255', '#661100', '#6699CC', '#888888']

# Figure sizing tuned for side-by-side LaTeX figures.
TEXT_WIDTH_IN = 6.5
FIG_WIDTH_IN = 0.45 * TEXT_WIDTH_IN
# Slightly taller aspect to avoid vertical compression at small widths.
FIG_HEIGHT_IN = 0.85*FIG_WIDTH_IN

# Scale typography with the figure width.
_BASE_FIG_WIDTH_IN = 6.0
_FONT_SCALE = 0.6
#FIG_WIDTH_IN / _BASE_FIG_WIDTH_IN
_TITLE_PAD = 4.0 * _FONT_SCALE
_PLOT_LINEWIDTH = 2.5 * _FONT_SCALE
matplotlib.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10.5 * _FONT_SCALE,
    'lines.linewidth': _PLOT_LINEWIDTH,
    'lines.markersize': 6.0 * _FONT_SCALE,
})

def certified_radius_plot(results: ResultDict, global_params, norm=Norm.L2):
    fig = plt.figure(dpi=72, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax = plt.gca()

    x_log = global_params.x_log
    max_radius = get_max_radius(results, norm) * 1.1

    if x_log:
        ax.set_xscale('log')
        min_radius = {Norm.LInf: 0.000001, Norm.L2: 0.00001, Norm.L1: 0.01}[norm]
        plot_radii = np.logspace(np.log10(min_radius), np.log10(max_radius), num=1000)
    else:
        plot_radii = np.linspace(0, max_radius, num=1000)

    labels = labels_dict[global_params.labels]
    plot_names = list(results.keys()) if global_params.original_name else list(labels.keys())

    # Exclude 'convex_noreg' from plot_names IMPORTANT
    plot_names = [name for name in plot_names if name != 'convex_noreg']

#     Diagnostic: report any requested names missing from results
#     missing = [name for name in plot_names if name not in results.keys()]
#     if missing:
#         print(f"[plot.py] Missing result keys for labels '{global_params.labels}': {missing}")
#         print(f"[plot.py] Available result keys: {sorted(results.keys())}")

    for name, color in zip(plot_names, line_colors):
        if name not in results.keys():
            continue

        # Get certified accuracies for both classes
        accs_0 = get_cert_accuracies(results, plot_radii, norm, target_class=0)[name]
        accs_1 = get_cert_accuracies(results, plot_radii, norm, target_class=1)[name]
        # Average (elementwise, up to min length)
        min_len = min(len(accs_0), len(accs_1))
        avg_accs = [(a0 + a1) / 2 for a0, a1 in zip(accs_0[:min_len], accs_1[:min_len])]
        avg_radii = plot_radii[:min_len]

        # Map labels: replace Convex* label with requested text
        if global_params.original_name:
            label_name = name
        else:
            label_name = labels.get(name, name)
            if 'Convex*' in label_name or name.startswith('convex_reg'):
                label_name = r'Ours'

        clean = get_clean_accuracy(results, name)

        # For abCROWN: only plot average at discrete evaluation levels (X markers).
        is_abcrown = (name == 'abcrown') or ('CROWN' in label_name.upper())
        if is_abcrown:
            # choose just your evaluation epsilons for this norm
            levels = _abcrown_levels(global_params, norm, max_radius)
            if levels:
                accs0 = get_cert_accuracies(results, levels, norm, target_class=0)[name]
                accs1 = get_cert_accuracies(results, levels, norm, target_class=1)[name]
                min_len = min(len(accs0), len(accs1))
                avg_levels = np.array(levels[:min_len])
                avg_accs   = np.array([(a0 + a1)/2 for a0, a1 in zip(accs0[:min_len], accs1[:min_len])])

                # Plot only those points as black X markers
                plt.plot(
                    avg_levels,
                    avg_accs,
                    linestyle='None',
                    marker='x',
                    markersize=6,
                    markeredgewidth=0.9,
                    markeredgecolor='k',   # force black markers
                    markerfacecolor='none',
                    alpha=0.95,
                    zorder=10,
                    label=f'{label_name} [{clean * 100:.1f}% clean] '
                )
            # No class curves for ab-CROWN
            continue

        # Plot Class 1 (dotted, black) (plot class1 first)
        plt.plot(plot_radii, accs_1, linestyle=':', color='k', alpha=0.8, linewidth=_PLOT_LINEWIDTH, label=f'{label_name}, Class 1')
        # Plot Class 2 (dashed, black)
        plt.plot(plot_radii, accs_0, linestyle='--', color='k', alpha=0.8, linewidth=_PLOT_LINEWIDTH, label=f'{label_name}, Class 2')
        # Plot Average (solid, black, thicker)
        plt.plot(avg_radii, avg_accs, linestyle='-', color='k', alpha=0.8, linewidth=_PLOT_LINEWIDTH, label=f'{label_name} [{clean * 100:.1f}% clean]')

    # Axis labels with capitalization and hyphen as requested
    plt.xlabel(f'{norm_str_dict[norm]}-Radius')
    plt.ylabel('Certified Accuracy')

    # Optional title for MNIST 3-8
    try:
        if getattr(global_params, "data", "") == "mnist_38":
            plt.title("MNIST 3-8", pad=2*_TITLE_PAD)
    except Exception:
        pass

    plt.legend(
        loc='upper right',
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.4,
        borderpad=0.3,
        markerscale=0.7,
        framealpha=0.95,
    )
    plt.xlim([min(plot_radii), max(plot_radii)])
    plt.ylim([0, 1])

    # Keep y-axis ticks unchanged (default). Customize x-axis ticks for linear scale.
    if not x_log:
        if norm == Norm.L2:
            step = 1.0
        elif norm == Norm.LInf:
            step = 0.1
        else:
            step = None

        if step is not None:
            max_x = max(plot_radii)
            ticks = np.arange(0.0, max_x + 1e-9, step)
            plt.xticks(ticks)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90)
    # Save as .pgf (vector) and .png (raster) using the explicit figure size.
    plt.savefig(figs_path(f'cert_{norm.name}_{global_params.labels}.pgf', global_params), transparent=True)
    plt.savefig(figs_path(f'cert_{norm.name}_{global_params.labels}.png', global_params), transparent=False, dpi=300)
    plt.close(fig)


def get_clean_accuracy(results: ResultDict, name: str) -> float:
    return np.mean([(r.target == r.pred).float().item() for r in results[name]])

def get_cert_accuracies(results: ResultDict, plot_radii: list[float], norm: Norm, target_class=0) -> dict[str, list[float]]:
    cert_accuracies = {}
    for (name, result_list) in results.items():
        result_list = filter_target_class(result_list, target_class)
        cert_radii = np.array(get_cert_radii(result_list, norm))
        cert_accuracies[name] = [np.mean(cert_radii >= thresh) for thresh in plot_radii]
    return cert_accuracies


def get_max_radius(results: ResultDict, norm=Norm.L2):
    # Flatten only list-like entries (ignore metadata like floats)
    flattened = []
    for v in results.values():
        if isinstance(v, list):
            flattened.extend(v)
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list):
                    flattened.extend(vv)

    result_list = filter_target_class(flattened)

    # compute maximum radius for given norm across filtered results
    max_r = 0.0
    for res in result_list:
        try:
            if getattr(res, "certificate", None) and getattr(res.certificate, "radius", None):
                rad = res.certificate.radius.get(norm, 0.0)
                max_r = max(max_r, float(rad))
        except Exception:
            continue

    return max_r


def get_cert_radii(result_list: list[Result], norm: Norm) -> list[float]:
    def get_cert(result: Result) -> Optional[Certificate]:
        return result.certificate

    def has_radius(result: Result) -> bool:
        return (result.target == result.pred).item() and (get_cert(result) is not None)

    return [get_cert(r).radius[norm] if has_radius(r) else -1 for r in result_list]

# Remove the filter for the target class, as we want to plot all classes
def filter_target_class(result_list, target_class: Optional[int] = None):
    """
    Normalize and filter `result_list` to a list of Result-like objects.
    Accepts a dict (will flatten list-valued entries) or an iterable of results.
    Ignores non-iterable metadata (floats/ints/None).
    """
    items = []

    # If a mapping/dict was passed, iterate its values
    if isinstance(result_list, dict):
        iterable = result_list.values()
    # If it's a list/tuple/set, accept it
    elif isinstance(result_list, (list, tuple, set)):
        iterable = result_list
    else:
        # Not a dict or iterable of results (e.g. float metadata) -> nothing to filter
        iterable = []

    for v in iterable:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for r in v:
                if hasattr(r, 'target'):
                    items.append(r)
            continue
        if hasattr(v, 'target'):
            items.append(v)
            continue
        # otherwise ignore (metadata like floats, dicts of metadata, etc.)
        continue

    if target_class is None:
        return items

    filtered = []
    for r in items:
        try:
            if int(getattr(r, "target")) == int(target_class):
                filtered.append(r)
        except Exception:
            continue
    return filtered

@click.command(context_settings={'show_default': True})
@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38')
@click.option('--experiment', type=click.Choice(main.experiments), default='standard')
@click.option('--clear_figs/--no_clear_figs', default=True)
@click.option('--labels', default='standard_1', help='Label set to use (missing entries ignored).')
@click.option('--original_name/--no_original_name', default=False)
@click.option('--x_log/--no_x_log', default=False)
def run(data, experiment, clear_figs, labels, original_name, x_log):
    pretty.init()

    pretty.section_print('Assembling parameters')
    experiment_directory = f'{data}-{experiment}'
    local_vars = locals()
    global_params = collections.namedtuple('Params', local_vars.keys())(*local_vars.values())

    file_utils.ensure_created_directory(f'./figs/{experiment_directory}', clear=clear_figs)

    pretty.section_print('Loading results')
    results: ResultDict = file_utils.read_pickle(dirs.out_path(experiment_directory, 'results.pkl'))

    pretty.section_print('Plotting results')
    certified_radius_plot(results, global_params, norm=Norm.L1)
    certified_radius_plot(results, global_params, norm=Norm.L2)
    certified_radius_plot(results, global_params, norm=Norm.LInf)


if __name__ == "__main__":
    run()
