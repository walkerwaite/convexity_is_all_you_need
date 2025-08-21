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

def certified_radius_plot(results: ResultDict, global_params, norm=Norm.L2):
    fig = plt.figure(dpi=72, figsize=(8, 6))
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
    plot_names = results.keys() if global_params.original_name else labels.keys()

    # Exclude 'convex_noreg' from plot_names IMPORTANT
    plot_names = [name for name in plot_names if name != 'convex_noreg']

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

        label_name = name if global_params.original_name else labels[name]
        clean = get_clean_accuracy(results, name)

        # Plot Class 2 (dotted, model color)
        plt.plot(plot_radii, accs_0, linestyle=':', color=color, alpha=0.8, label=f'{label_name} Class 2')
        # Plot Class 1 (dashed, model color)
        plt.plot(plot_radii, accs_1, linestyle='--', color=color, alpha=0.8, label=f'{label_name} Class 1')
        # Plot Average (solid, model color, thicker)
        plt.plot(avg_radii, avg_accs, linestyle='-', color=color, alpha=0.8, linewidth=2, label=f'{label_name} Average [{clean * 100:.1f}% clean]')

    plt.xlabel(f'{norm_str_dict[norm]} radius')
    plt.ylabel('Certified accuracy')
    plt.legend(loc='upper right', handlelength=1, handletextpad=0.3)
    plt.xlim([min(plot_radii), max(plot_radii)])
    plt.ylim([0, 1])
    fig.tight_layout()
    # Save as .pgf
    plt.savefig(figs_path(f'cert_{norm.name}_{global_params.labels}.pgf', global_params), transparent=True, bbox_inches='tight')
    # Save as .png
    plt.savefig(figs_path(f'cert_{norm.name}_{global_params.labels}.png', global_params), transparent=False, bbox_inches='tight')
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


def get_max_radius(results: ResultDict, norm: Norm) -> float:
    result_list = filter_target_class(sum(results.values(), []))
    cert_radii = get_cert_radii(result_list, norm)
    return max(cert_radii)


def get_cert_radii(result_list: list[Result], norm: Norm) -> list[float]:
    def get_cert(result: Result) -> Optional[Certificate]:
        return result.certificate

    def has_radius(result: Result) -> bool:
        return (result.target == result.pred).item() and (get_cert(result) is not None)

    return [get_cert(r).radius[norm] if has_radius(r) else -1 for r in result_list]

# Remove the filter for the target class, as we want to plot all classes
def filter_target_class(result_list: list[Result], target_class=0) -> list[Result]:
    return [r for r in result_list if int(r.target) == target_class]
    # target class == 0

@click.command(context_settings={'show_default': True})
@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38')
@click.option('--experiment', type=click.Choice(main.experiments), default='standard')
@click.option('--clear_figs/--no_clear_figs', default=True)
@click.option('--labels', type=click.Choice(labels_dict.keys()), default='standard_1')
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
