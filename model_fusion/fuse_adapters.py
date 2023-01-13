
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on
GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import logging
import os
from shutil import copyfile
import json
import torch
import sys
# todo package for otfusion
import otfusion.wasserstein_ensemble as wasserstein_ensemble
import copy
import otfusion.utils as utils
import otfusion.activation_utils as au
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
sys.path = ['/user/sunsiqi/hs/ModelFusion/otfusion', '/user/sunsiqi/hs/MoE/adapter-transformers-master/src', '/user/sunsiqi/hs/MoE/adapter-transformers-master/src/transformers',
            '/user/sunsiqi/.pycharm_helpers/pydev', '/user/sunsiqi/.pycharm_helpers/pycharm_display', '/user/sunsiqi/.pycharm_helpers/third_party/thriftpy',
            '/Users/Lenovo/AppData/Local/JetBrains/PyCharm2021.2/cythonExtensions', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python38.zip', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/lib-dynload', '/user/sunsiqi/.local/lib/python3.8/site-packages', '/user/sunsiqi/.local/lib/python3.8/site-packages/pdbx-1.0-py3.8.egg',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/site-packages', '/user/sunsiqi/.pycharm_helpers/pycharm_matplotlib_backend']

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

from dataclasses import dataclass, field
from typing import Optional
from transformers.adapters.configuration import PfeifferConfig
from tqdm import tqdm
from transformers import (
    AdapterArguments,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    AutoTokenizer,
    default_data_collator,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

# todo avg
def Average(root, adapters, Reduce=16, name='avg'):

    params = {}
    dicts = 0
    for adapter in tqdm(adapters):
        adapter_dir = os.path.join(root, adapter)
        adapter_config = os.path.join(adapter_dir, 'adapter_config.json')
        with open(adapter_config, 'r') as f:
            adapter_config = json.load(f)
        adapter_name = adapter_config["name"]
        config = adapter_config["config"]
        reduce = config["reduction_factor"]
        # todo adapter temporally, ignoring the head.
        if reduce == Reduce:
            state_dict = torch.load(os.path.join(adapter_dir, 'pytorch_adapter.bin'))
            # todo rename the keys
            for key in state_dict.keys():
                new_key = key.replace(adapter_name, 'avg')
                params[new_key] = state_dict[key] if new_key not in params else \
                    params[new_key] + state_dict[key]
            dicts += 1

    # todo avg
    for key in params:
        params[key] = params[key] / dicts

    # todo save fused adapter
    tgt_root = os.path.join(root, name)
    if not os.path.exists(tgt_root):
        os.mkdir(tgt_root)
    copyfile(os.path.join(adapter_dir, 'head_config.json'), os.path.join(tgt_root, 'head_config.json'))
    copyfile(os.path.join(adapter_dir, 'pytorch_model_head.bin'), os.path.join(tgt_root, 'pytorch_model_head.bin'))
    with open(os.path.join(tgt_root, 'adapter_config.json'), 'w') as f:
        adapter_config["name"] = name
        json.dump(adapter_config, f, indent=2, sort_keys=True)
    torch.save(params, os.path.join(tgt_root, 'pytorch_adapter.bin'))

# todo sum
def Summation(root, adapters, Reduce=16, name='sum'):

    params = {}
    dicts = 0
    for adapter in tqdm(adapters):
        adapter_dir = os.path.join(root, adapter)
        adapter_config = os.path.join(adapter_dir, 'adapter_config.json')
        with open(adapter_config, 'r') as f:
            adapter_config = json.load(f)
        adapter_name = adapter_config["name"]
        config = adapter_config["config"]
        reduce = config["reduction_factor"]
        # todo adapter temporally, ignoring the head.
        if reduce == Reduce:
            state_dict = torch.load(os.path.join(adapter_dir, 'pytorch_adapter.bin'))
            # todo rename the keys
            for key in state_dict.keys():
                new_key = key.replace(adapter_name, 'avg')
                params[new_key] = state_dict[key] if new_key not in params else \
                    params[new_key] + state_dict[key]
            dicts += 1

    # todo save fused adapter
    tgt_root = os.path.join(root, name)
    if not os.path.exists(tgt_root):
        os.mkdir(tgt_root)
    copyfile(os.path.join(adapter_dir, 'head_config.json'), os.path.join(tgt_root, 'head_config.json'))
    copyfile(os.path.join(adapter_dir, 'pytorch_model_head.bin'), os.path.join(tgt_root, 'pytorch_model_head.bin'))
    with open(os.path.join(tgt_root, 'adapter_config.json'), 'w') as f:
        adapter_config["name"] = name
        json.dump(adapter_config, f, indent=2, sort_keys=True)
    torch.save(params, os.path.join(tgt_root, 'pytorch_adapter.bin'))


def save_adapter(params, adapter_config, adapter_dir, root, name):
    # todo save fused adapter
    tgt_root = os.path.join(root, name)
    if not os.path.exists(tgt_root):
        os.mkdir(tgt_root)
    copyfile(os.path.join(adapter_dir, 'head_config.json'), os.path.join(tgt_root, 'head_config.json'))
    copyfile(os.path.join(adapter_dir, 'pytorch_model_head.bin'), os.path.join(tgt_root, 'pytorch_model_head.bin'))
    with open(os.path.join(tgt_root, 'adapter_config.json'), 'w') as f:
        adapter_config["name"] = name
        json.dump(adapter_config, f, indent=2, sort_keys=True)
    torch.save(params, os.path.join(tgt_root, 'pytorch_adapter.bin'))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            " Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    data_collator = default_data_collator
    if adapter_args.geom_ensemble_type == 'acts':
        train_loader = DataLoader(
                    train_dataset,
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory,
                )
    else:
        train_loader = None

    # TODO Load model
    root = '/user/sunsiqi/hs/MoE/adapter-transformers-master/adapters'
    adapter_names = []
    adapters = [
        "sentiment/sst-2@ukp",
        "nli/multinli@ukp",
        "nli/rte@ukp",
        "sts/mrpc@ukp",
        "sts/qqp@ukp",
        "comsense/cosmosqa@ukp",
        "comsense/csqa@ukp",
        "comsense/hellaswag@ukp",
        "comsense/siqa@ukp",
        "comsense/winogrande@ukp",
        "nli/cb@ukp",
        "nli/sick@ukp",
        "nli/scitail@ukp",
        "qa/boolq@ukp",
        "sentiment/imdb@ukp",
    ]
    # todo prepare the pretrained adapters for avg and sum.
    for elem in tqdm(adapters):
        # try:
        left = elem.split('/')[0]
        if not os.path.exists(os.path.join(root, left)):
            os.mkdir(os.path.join(root, left))
        # todo download the adapters when local adapters are removed mistakenly.
        # name = model.load_adapter(elem, config=PfeifferConfig(), with_head=False)
        # model.save_adapter(os.path.join(root, elem), name)
        name = model.load_adapter(os.path.join(root, elem), config=PfeifferConfig(), with_head=False)
        adapter_names.append(name)
        if not os.path.exists(os.path.join(root, elem)):
            model.save_adapter(os.path.join(root, elem), name)
            model.delete_adapter(name)
        else:
            model.delete_adapter(name)
            continue

    # todo average or sum the adapters
    # Average(root, adapters)
    # Summation(root, adapters)

    # todo otfusion
    adapter_args.gpu_id = int(training_args.device_ids.split(' ')[0])
    adapter_args.gpu_id = -1
    elem = adapters.pop()
    model1 = copy.deepcopy(model)
    model = copy.deepcopy(model1)
    adapter_dir = '/user/sunsiqi/hs/MoE/adapter-transformers-master/adapters/nli/sick@ukp'
    previous_name = model.load_adapter(os.path.join(root, elem), config=PfeifferConfig(), with_head=False)
    model.set_active_adapters([previous_name])
    adapter_config = os.path.join(adapter_dir, 'adapter_config.json')
    with open(adapter_config, 'r') as f:
        adapter_config = json.load(f)

    # todo fuse one by one.
    while len(adapters) > 0:
        elem = adapters.pop()
        name = model1.load_adapter(os.path.join(root, elem), config=PfeifferConfig(), with_head=False)
        model1.set_active_adapters([name])
        print("Now fuse {} and {}.".format(previous_name, name))
        activations = utils.get_model_activations(adapter_args, [model, model1], train_loader) \
                            if adapter_args.geom_ensemble_type == 'acts' else None
        params = wasserstein_ensemble.geometric_ensembling_modularized(adapter_args, [model, model1], activations)
        params = dict((k.replace(previous_name, previous_name + '-' + name), v) for k, v in params.items())
        # todo delete previous adapters.
        model1.delete_adapter(name)
        model.delete_adapter(previous_name)
        previous_name = previous_name + '-' + name
        # todo load the fused adapters prepared for the next step.
        save_adapter(params, adapter_config, adapter_dir, root + '/otfusion', previous_name)
        model.load_adapter(os.path.join(root + '/otfusion', previous_name), config=PfeifferConfig(), with_head=False)
        model.set_active_adapters([previous_name])


if __name__ == "__main__":
    main()
