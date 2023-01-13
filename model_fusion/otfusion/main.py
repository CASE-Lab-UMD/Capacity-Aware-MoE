import parameters
from data import get_dataloader
import routines
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys
sys.path = ['/user/sunsiqi/hs/MoE/adapter-transformers-master/src', '/user/sunsiqi/hs/MoE/adapter-transformers-master/src/transformers',
            '/user/sunsiqi/.pycharm_helpers/pydev', '/user/sunsiqi/.pycharm_helpers/pycharm_display', '/user/sunsiqi/.pycharm_helpers/third_party/thriftpy',
            '/Users/Lenovo/AppData/Local/JetBrains/PyCharm2021.2/cythonExtensions', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python38.zip', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/lib-dynload', '/user/sunsiqi/.local/lib/python3.8/site-packages', '/user/sunsiqi/.local/lib/python3.8/site-packages/pdbx-1.0-py3.8.egg',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/site-packages', '/user/sunsiqi/.pycharm_helpers/pycharm_matplotlib_backend']

import torch
import time
from dataclasses import dataclass, field
from typing import Optional
from transformers.adapters.configuration import PfeifferConfig


from tqdm import tqdm
from transformers import (
    AdapterArguments,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)


def main():

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # todo loading configuration
    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config
    # todo obtain pretrained models
    if args.load_models != '':
        print("------- Loading pre-trained models -------")
        ensemble_experiment = args.load_models.split('/')
        if len(ensemble_experiment) > 1:
            # both the path and name of the experiment have been specified
            ensemble_dir = args.load_models
        elif len(ensemble_experiment) == 1:
            # otherwise append the directory before!
            ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
            ensemble_dir = ensemble_root_dir + args.load_models

        if args.dataset == 'mnist':
            train_loader, test_loader = get_dataloader(args)
            retrain_loader, _ = get_dataloader(args)

        models = []
        accuracies = []

        for idx in range(args.num_models):
            print("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))
            if args.dataset.lower()[0:7] == 'cifar10' and (
                    args.model_name.lower()[0:5] == 'vgg11' or args.model_name.lower()[0:6] == 'resnet'):
                if idx == 0:
                    config_used = config
                elif idx == 1:
                    config_used = second_config

                model, accuracy = cifar_train.get_pretrained_model(
                    config_used, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)),
                    args.gpu_id, relu_inplace=not args.prelu_acts
                    # if you want pre-relu acts, set relu_inplace to False
                )
            else:
                model, accuracy = routines.get_pretrained_model(
                    args, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)), idx=idx
                )
            models.append(model)
            accuracies.append(accuracy)
        print("Done loading all the models")

    if args.same_model != -1:
        print("Debugging with same model")
        model, acc = models[args.same_model], accuracies[args.same_model]
        models = [model, model]
        accuracies = [acc, acc]

    # todo second_config is not needed here as well, since it's just used for the dataloader!

    print("Activation Timer start")

    st_time = time.perf_counter()
    activations = utils.get_model_activations(args, models, config=config)
    end_time = time.perf_counter()
    setattr(args, 'activation_time', end_time - st_time)
    print("Activation Timer ends")

    for idx, model in enumerate(models):
        setattr(args, f'params_model_{idx}', utils.get_model_size(model))

    # todo set seed for numpy based calculations
    NUMPY_SEED = 100
    np.random.seed(NUMPY_SEED)

    # todo run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    # Deprecated: wasserstein_ensemble.geometric_ensembling(models, train_loader, test_loader)
    print("Timer start")
    st_time = time.perf_counter()
    geometric_acc, geometric_model, unmerged_models_layers \
        = wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader, test_loader, activations)
    # todo added by zz
    # merge_by_various_weight(unmerged_models_layers, test_loader)

    end_time = time.perf_counter()
    print("Timer ends")
    setattr(args, 'geometric_time', end_time - st_time)
    args.params_geometric = utils.get_model_size(geometric_model)
    print("Time taken for geometric ensembling is {} seconds".format(str(end_time - st_time)))

if __name__ == '__main__':
    main()



