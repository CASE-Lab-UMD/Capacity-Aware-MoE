


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


import dataclasses
import logging
import os
import sys
from datasets import load_dataset, load_metric

sys.path = ['/user/sunsiqi/hs/MoE/adapter-transformers-master/src', '/user/sunsiqi/hs/MoE/adapter-transformers-master/src/transformers',
            '/user/sunsiqi/.pycharm_helpers/pydev', '/user/sunsiqi/.pycharm_helpers/pycharm_display', '/user/sunsiqi/.pycharm_helpers/third_party/thriftpy',
            '/Users/Lenovo/AppData/Local/JetBrains/PyCharm2021.2/cythonExtensions', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python38.zip', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/lib-dynload', '/user/sunsiqi/.local/lib/python3.8/site-packages', '/user/sunsiqi/.local/lib/python3.8/site-packages/pdbx-1.0-py3.8.egg', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/site-packages', '/user/sunsiqi/.pycharm_helpers/pycharm_matplotlib_backend']

from dataclasses import dataclass, field
from typing import Dict, Optional
from transformers.adapters.configuration import PfeifferConfig

import numpy as np
from tqdm import tqdm
from transformers import (
    AdapterArguments,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


logger = logging.getLogger(__name__)


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

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # TODO Load model
    root = '/user/sunsiqi/hs/MoE/adapter-transformers-master/adapters'
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
        "sentiment/imdb@ukp"
    ]
    for elem in tqdm(adapters):
        left = elem.split('/')[0]
        if not os.path.exists(os.path.join(root, left)):
            os.mkdir(os.path.join(root, left))
        name = model.load_adapter(os.path.join(root, elem), config=PfeifferConfig(), with_head=False)
        # model.save_adapter(os.path.join(root, elem), name)

    # save model
    # model.save_pretrained('./path/to/model/directory/')# save adapter
    # model.save_adapter(' ./path/to/adapter/directory/', 'sst-2')

if __name__ == "__main__":
    main()
