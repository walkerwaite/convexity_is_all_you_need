from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Surpress tensorflow cuda errors
os.environ['GIT_PYTHON_REFRESH'] = 'quiet' # Suppress git import errors

import warnings
# Surpress lightning-bolt warnings https://github.com/Lightning-Universe/lightning-bolts/issues/563
warnings.simplefilter('ignore')
original_filterwarnings = warnings.filterwarnings
def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append':True})
warnings.filterwarnings = _filterwarnings

import random
import collections
import click
import numpy as np
from dataclasses import dataclass
import dacite

import torch
from pytorch_lightning import LightningDataModule

from convexrobust.data import datamodules

# Import activation functions
from convexrobust.model.modules import SmeLU, nn  

# from convexrobust.model.linf_certifiable import LInfCertifiable
from convexrobust.model.randsmooth_certifiable import RandsmoothCertifiable
from convexrobust.model.insts.convex import (
    ConvexCifar, ConvexMnist, ConvexFashionMnist, ConvexMalimg, ConvexSimple
)
from convexrobust.model.insts.randsmooth import (
    RandsmoothCifar, RandsmoothMnist, RandsmoothFashionMnist,RandsmoothMalimg, RandsmoothSimple
)
from convexrobust.model.insts.cayley import (
    CayleyCifar, CayleyMnist, CayleyFashionMnist
)
from convexrobust.model.insts.abcrown import (
    ABCROWNCifar, ABCROWNMnist, ABCROWNFashionMnist, ABCROWNMalimg
)

from convexrobust.utils import dirs, pretty, file_utils
from convexrobust.utils import file_utils, torch_utils as TU
from convexrobust.model.certificate import Norm


from convexrobust.main.train import TrainConfig
from convexrobust.main.evaluate import EvaluateConfig
import convexrobust.main.train as main_train
import convexrobust.main.evaluate as main_evaluate
from convexrobust.main.train import ModelBlueprint, BlueprintDict, ModelDict

from typing import Optional

experiments = ['standard', 'ablation', 'ablation_noaugment']


def randsmooth_blueprints(
        randsmooth_class: type[RandsmoothCertifiable], epochs: int, sigma_scale: float,
        data_in_n: int, nb=100, large_splitderandomized_sigma=None, load=False
    ) -> BlueprintDict:

    constructor_params = {'n': 10000, 'cert_n_scale': 10, 'nb': nb, 'data_in_n': data_in_n}


    blueprint_params = {'load_model': True, 'load_eval_results': True} if load else {}

    blueprints = {}

    for factor in [1]:
        params = {'sigma': sigma_scale * factor, **constructor_params}
        blueprints.update({
            f'randsmooth_splitderandomized_{factor}': ModelBlueprint(
                randsmooth_class(noise='split_derandomized', **params),
                epochs * factor, **blueprint_params
            ),
            f'randsmooth_laplace_{factor}': ModelBlueprint(
                randsmooth_class(noise='laplace', **params), epochs * factor, **blueprint_params
            ),
            f'randsmooth_gauss_{factor}': ModelBlueprint(
                randsmooth_class(noise='gauss', **params), epochs * factor, **blueprint_params
            ),
            f'randsmooth_uniform_{factor}': ModelBlueprint(
                randsmooth_class(noise='uniform', **params), epochs * factor, **blueprint_params
            ),
        })

    if large_splitderandomized_sigma is not None:
        params = {'sigma': large_splitderandomized_sigma, **constructor_params}
        blueprints[f'randsmooth_splitderandomized_large'] = ModelBlueprint(
            randsmooth_class(noise='split_derandomized', **params), epochs * 4, **blueprint_params
        )

    return blueprints


def get_blueprints(datamodule: LightningDataModule, experiment: str, activation_function) -> BlueprintDict:
    if experiment in ['ablation', 'ablation_noaugment']:
        assert datamodule.name in ['cifar10_catsdogs', 'cifar10_dogscats']

    data_args = {'data_in_n': datamodule.in_n}

    if datamodule.name == 'mnist_38':
        return {
            #'convex_noreg': ModelBlueprint(ConvexMnist(nonlin=activation_function), 150),
            'convex_reg': ModelBlueprint(ConvexMnist(nonlin=activation_function, reg=0.01), 150),
            # Commented out Cayley and Randsmooth models
            # 'cayley': ModelBlueprint(CayleyMnist(**data_args), 60),
            'abcrown': ModelBlueprint(ABCROWNMnist(), 60),
            # **randsmooth_blueprints(RandsmoothMnist, 60, 0.75, datamodule.in_n),
        }
    elif datamodule.name == 'fashion_mnist_shirts':
        return {
            # 'convex_noreg': ModelBlueprint(ConvexFashionMnist(nonlin=activation_function), 60),
            'convex_reg': ModelBlueprint(ConvexFashionMnist(nonlin=activation_function, reg=0.01), 60),
            # Commented out Cayley and Randsmooth models
            # 'cayley': ModelBlueprint(CayleyFashionMnist(**data_args), 60),
            'abcrown': ModelBlueprint(ABCROWNFashionMnist(), 60),
            # **randsmooth_blueprints(RandsmoothFashionMnist, 60, 0.75, datamodule.in_n),
        }
    elif datamodule.name == 'malimg':
        return {
            # 'convex_noreg': ModelBlueprint(ConvexMalimg(nonlin=activation_function, reg=0.0), 150),
            'convex_reg': ModelBlueprint(ConvexMalimg(nonlin=activation_function, reg=0.075), 150),
            # Commented out Randsmooth models
            'abcrown': ModelBlueprint(ABCROWNMalimg(), 150),
            # **randsmooth_blueprints(
            #     RandsmoothMalimg, 150, 3.5, datamodule.in_n, nb=32,
            #     large_splitderandomized_sigma=100
            # ),
        }
    elif datamodule.name in ['cifar10_catsdogs', 'cifar10_dogscats']:
        if experiment == 'standard':
            return {
                # 'convex_noreg': ModelBlueprint(ConvexCifar(nonlin=activation_function), 150),
                'convex_reg': ModelBlueprint(ConvexCifar(nonlin=activation_function, reg=0.0075), 150),
                # Commented out Cayley and Randsmooth models
                # 'cayley': ModelBlueprint(CayleyCifar(**data_args), 150),
                'abcrown': ModelBlueprint(ABCROWNCifar(), 150),
                # **randsmooth_blueprints(RandsmoothCifar, 600, 0.75, datamodule.in_n),
            }
        elif experiment == 'ablation':
            # Experiments for appendix G.2
            blueprints = {
                'convex_nofeaturemap': ModelBlueprint(
                    ConvexCifar(apply_feature_map=False, reg=0.0), 150
                )
            }

            for i, reg in enumerate([0.0, 0.0025, 0.005, 0.0075, 0.01]):
                blueprints[f'convex_reg_{i}'] = ModelBlueprint(
                    ConvexCifar(reg=reg), 150
                )

            return blueprints
        elif experiment == 'ablation_noaugment':
            # Experiments for appendix G.3
            return {
                'convex_nofeaturemap': ModelBlueprint(
                    ConvexCifar(apply_feature_map=False), 500
                )
            }
        else:
            raise RuntimeError('Bad experiment type')
    elif datamodule.name == 'circles':
        return {
            'convex_noreg': ModelBlueprint(ConvexSimple(), 30)
        }
    else:
        raise RuntimeError('Bad dataset')


@click.command(context_settings={'show_default': True})

@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38', help="""
The dataset to use. All should be downloaded automatically.
""")
@click.option('--experiment', type=click.Choice(experiments), default='standard', help="""
Which experiment to run (e.g. standard, ablation, etc).
""")
@click.option('--clear/--no_clear', default=False, help="""
Whether to clear all old models and results and start fresh.
""")
@click.option('--tensorboard/--no_tensorboard', default=True, help="""
Whether to launch tensorboard showing the results of training. Works even with no_train.
""")
@click.option('--seed', default=1, help="""
The random seed.
""")

# Activation function options
@click.option('--activation', required=False, type=str, default='ReLU', help='Activation function to use (SmeLU or ReLU).')

@click.option('--train/--no_train', default=False, help="""
Whether to train models or load them from a previous training. You can retrain only
certain models by specifying --train and fixing the other models with the load_model
flag in ModelBlueprint.
""")
@click.option('--augment_data/--no_augment_data', default=True, help="""
Whether to apply data augmentation in the datamodule (e.g. random cropping).
Does NOT affect noise augmentation for randomized smoothing methods.
""")
@click.option('--balance/--no_balance', default=True, help="""
Whether to balance the test set performance of methods after training such that
the accuracies are the same across both classes.
""")

@click.option('--eval_n', default=1000, help="""
How many test data points to evaluate / balance over.
""")
@click.option('--verify_cert/--no_verify_cert', default=False, help="""
Whether to verify any certificates generated during evaluation with a PGD attack.
Does not verify nondeterministic randomized smoothing certificates.
""")
@click.option('--empirical_cert/--no_empirical_cert', default=False, help="""
Whether to compute empirical robustness certificates with a PGD attack.
""")
def run(
        data, experiment, clear, tensorboard, seed,
        train, activation, augment_data, balance,
        eval_n, verify_cert, empirical_cert
    ) -> None:
    
    # Map activation function string to the actual class
    activation_map = {
        'ReLU': nn.ReLU,
        'SmeLU': SmeLU
    }
    if activation not in activation_map:
        raise ValueError(f"Invalid activation function: {activation}. Choose from {list(activation_map.keys())}.")

    activation_function = activation_map[activation]

    assert not (clear and (not train)) # If clear old models, must train!
    init(seed)

    pretty.section_print('Loading data and assembling parameters')
    params = locals() # Combine args + datamodule and experiment_directory attributes

    train_config = dacite.from_dict(data_class=TrainConfig, data=params)
    evaluate_config = dacite.from_dict(data_class=EvaluateConfig, data=params)

    datamodule = datamodules.get_datamodule(data, eval_n=eval_n, no_transforms=not augment_data)
    experiment_directory = f'{data}-{experiment}'

    if clear:
        file_utils.create_empty_directory(dirs.out_path(experiment_directory))
    if tensorboard:
        TU.launch_tensorboard(dirs.out_path(experiment_directory), 6006)

    blueprints: BlueprintDict = get_blueprints(datamodule, experiment, activation_function)

    pretty.section_print('Creating models')
    models: ModelDict = main_train.train_models(
        blueprints, experiment_directory, datamodule, train_config
    )

    pretty.section_print('Evaluating models')
    _ = main_evaluate.evaluate_models(
        models, blueprints, experiment_directory, datamodule, evaluate_config
    )
    
    # --- Print random 10 certified radii and clean accuracy for each class ---

    results_path = dirs.out_path(f'{experiment_directory}', 'results.pkl')
    results = file_utils.read_pickle(results_path)

    for model_name, result_list in results.items():
        print(f"\nModel: {model_name}")

        for class_label in [0, 1]:
            # Filter results for this class
            class_results = [r for r in result_list if int(r.target) == class_label and r.certificate is not None]

            # Print clean accuracy for this class REGARDLESS OF CERTIFICATE!!!
            class_results_all = [r for r in result_list if int(r.target) == class_label]
            clean_acc = sum(int(r.target == r.pred) for r in class_results_all) / max(1, len(class_results_all))
            print(f"  Class {class_label} clean accuracy: {clean_acc * 100:.2f}%")

            # Get all certified radii for L1, L2, LInf for this class (ONLY IF CERTIFICATE EXISTS)
            class_results_cert = [r for r in class_results_all if r.certificate is not None]
            radii = [
                (float(r.certificate.radius[Norm.L1]), float(r.certificate.radius[Norm.L2]), float(r.certificate.radius[Norm.LInf]))
                for r in class_results_cert if hasattr(r.certificate, 'radius')
            ]
            sample_radii = random.sample(radii, min(10, len(radii))) if radii else []

            print(f"  Sample certified radii (L1, L2, LInf) for class {class_label}:")
            for rad in sample_radii:
                print(f"    L1: {rad[0]:.4f}, L2: {rad[1]:.4f}, LInf: {rad[2]:.4f}")
                
    pretty.section_print('Done executing (ctrl+c to exit)')

def init(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    pretty.init()


if __name__ == "__main__":
    run()
