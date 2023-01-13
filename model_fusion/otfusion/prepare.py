import torch
import numpy as np
import otfusion.utils_zz as uz
from otfusion.data import get_dataloader
import otfusion.routines as routines
from datasets import load_dataset, load_metric
from torch import nn
import copy
import sys
sys.path = ['/user/sunsiqi/hs/MoE/adapter-transformers-master/src', '/user/sunsiqi/hs/MoE/adapter-transformers-master/src/transformers',
            '/user/sunsiqi/.pycharm_helpers/pydev', '/user/sunsiqi/.pycharm_helpers/pycharm_display', '/user/sunsiqi/.pycharm_helpers/third_party/thriftpy',
            '/Users/Lenovo/AppData/Local/JetBrains/PyCharm2021.2/cythonExtensions', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python38.zip', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/lib-dynload', '/user/sunsiqi/.local/lib/python3.8/site-packages', '/user/sunsiqi/.local/lib/python3.8/site-packages/pdbx-1.0-py3.8.egg', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/site-packages', '/user/sunsiqi/.pycharm_helpers/pycharm_matplotlib_backend']

from transformers import MBartModel, AutoConfig
# import utils_zz2 as uz2
# import transformers
# from fairseq.models.transformer import TransformerModel


def mnist_test(version=1):
    import utils_zz2 as uz2
    mlp_args = {
        'dataset': 'mnist',
        'model_name': 'mlpnet',
        "num_hidden_nodes1": 400,
        "num_hidden_nodes2": 200,
        "num_hidden_nodes3": 100,
        "enable_dropout": False
    }
    args = uz.get_args(**mlp_args)
    # args_modify(args)
    # args.dataset = "dummy"
    # dummy_train_loader, dummy_test_loader = get_dataloader(args, unit_batch=True)
    train_loader, test_loader = get_dataloader(args, root_dir=ROOT)

    model0, accuracy0 = routines.get_pretrained_model(args, f'{ROOT}/mnist_models/model_0/final.checkpoint')
    model1, accuracy1 = routines.get_pretrained_model(args, f'{ROOT}/mnist_models/model_1/final.checkpoint')
    
    # activations_origin = ca.compute_activations_across_models_v1(args, models, train_loader, args.act_num_samples, mode=args.activation_mode)
    ## 用 dummy train loader 效果不好
    # activations = uz.compute_activations_across_models_zz(args, models, dummy_train_loader, args.act_num_samples, mode=args.activation_mode)
    acts_inputs = uz.get_inputs_for_acts(args, train_loader)
    if version == 1:
        activations = uz.compute_activations_across_models_zz(args, [model0, model1], acts_inputs)
    elif version == 2:
        activations = uz2.compute_activations_across_models_zz(args, [model0, model1], acts_inputs)
    elif version == 3:
        import activation_utils
        activations = activation_utils.ActivationsManager(args, [model0, model1], acts_inputs)
    
    print(f"accuracy0: {accuracy0};  accuracy1: {accuracy1}")
    return args, model0, model1, activations, train_loader, test_loader, acts_inputs

def cifar_vgg_test(version=1):
    import utils_zz2 as uz2
    mlp_args = {
        'dataset': 'Cifar10',
        'model_name': 'vgg11_nobias'
    }
    args = uz.get_args(**mlp_args)
    # args_modify(args)
    # args.dataset = "dummy"
    # dummy_train_loader, dummy_test_loader = get_dataloader(args, unit_batch=True)
    train_loader, test_loader = get_dataloader(args, root_dir=ROOT)


    model0, accuracy0 = routines.get_pretrained_model(args, f'{ROOT}/cifar_models/model_0/best.checkpoint')
    model1, accuracy1 = routines.get_pretrained_model(args, f'{ROOT}/cifar_models/model_1/best.checkpoint')
    print(f"accuracy0: {accuracy0};  accuracy1: {accuracy1}")

    # activations_origin = ca.compute_activations_across_models_v1(args, models, train_loader, args.act_num_samples, mode=args.activation_mode)
    ## 用 dummy train loader 效果不好
    # activations = uz.compute_activations_across_models_zz(args, models, dummy_train_loader, args.act_num_samples, mode=args.activation_mode)
    acts_inputs = uz.get_inputs_for_acts(args, train_loader)
    if version == 1:
        activations = uz.compute_activations_across_models_zz(args, [model0, model1], acts_inputs)
    elif version == 2:
        activations = uz2.compute_activations_across_models_zz(args, [model0, model1], acts_inputs)

    return args, model0, model1, activations,  train_loader, test_loader, acts_inputs

def cifar_resnet_test(version=1):
    import utils_zz2 as uz2

    mlp_args = {
        'dataset': 'Cifar10',
        'model_name': 'resnet18_nobias_nobn'
    }
    args = uz.get_args(**mlp_args)
    # args_modify(args)
    # args.dataset = "dummy"
    # dummy_train_loader, dummy_test_loader = get_dataloader(args, unit_batch=True)
    train_loader, test_loader = get_dataloader(args, root_dir=ROOT)


    model0, accuracy0 = routines.get_pretrained_model(args, f'{ROOT}/resnet_models/model_0/best.checkpoint')
    model1, accuracy1 = routines.get_pretrained_model(args, f'{ROOT}/resnet_models/model_1/best.checkpoint')
    print(f"accuracy0: {accuracy0};  accuracy1: {accuracy1}")

    # activations_origin = ca.compute_activations_across_models_v1(args, models, train_loader, args.act_num_samples, mode=args.activation_mode)
    ## 用 dummy train loader 效果不好
    # activations = uz.compute_activations_across_models_zz(args, models, dummy_train_loader, args.act_num_samples, mode=args.activation_mode)
    acts_inputs = uz.get_inputs_for_acts(args, train_loader)

    if version == 1:
        activations = uz.compute_activations_across_models_zz(args, [model0, model1], acts_inputs)
    elif version == 2:
        activations = uz2.compute_activations_across_models_zz(args, [model0, model1], acts_inputs)

    return args, model0, model1, activations, train_loader, test_loader, acts_inputs

def test_mlm(model, tokenizer, head=None, text="Hello I'm a [MASK] model."):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    # head = BertOnlyMLMHead(model_config)
    if head == None:
        prediction_scores = output['logits']
    else:
        prediction_scores = head(output[0])
    print(tokenizer.decode(prediction_scores.squeeze().argmax(axis=1)))
    # encoded_input
    # encoded_input['input_ids'], prediction_scores.squeeze().argmax(axis=1)

def glue_param_load(model: nn.Module, task, tag='bert-base-uncased', model_name="pytorch_model", omit_classifier=True, seed='1'):
    # model_name_or_path = f'/public/data0/NLP/zhangzheng15/transformers/checkpoints/{tag}/saved_models/{task}/{model_name}{seed}/pytorch_model.bin'
    model_name_or_path = f'/public/data0/NLP/zhangzheng15/transformers/checkpoints/{tag}/pytorch_model/{task}/pytorch_model.bin'
    
    checkpoint = torch.load(model_name_or_path, map_location=torch.device('cpu'))
    if omit_classifier:
        omit_layers = [layer for layer in checkpoint.keys() if "classifier" in layer]
        for layer in omit_layers:
            del checkpoint[layer]
    model.load_state_dict(checkpoint, strict=False)
    
def glue_ft_load(task, tag='bert-base-uncased', max_seq_length=128, batch_size=256, seed=''):
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EvalPrediction,
        default_data_collator,
        set_seed
    )
    assert task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
    # model_name_or_path = f'/public/data0/NLP/zhangzheng15/transformers/checkpoints/{tag}/pytorch_model/{task}{seed}'
    model_name_or_path = f'/workspace/data/users/zhangzheng15/transformers/checkpoints/{tag}/saved_models/{task}/pytorch_model{seed}'

    cache_dir = '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/code/transformers/datasets/hf-cached'
    from datasets import load_dataset, load_metric
    raw_datasets = load_dataset(
            '/workspace/data/users/zhangzheng15/transformers/datasets/hf-cached/glue.py', task, cache_dir=cache_dir
        )
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=None,
        cache_dir=cache_dir,
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        revision='main',
        use_auth_token=None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=cache_dir,
        revision='main',
        use_auth_token=None,
    )
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
    sentence1_key, sentence2_key = task_to_keys[task]

    padding = "max_length"
    if not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
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
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation_matched" if task == "mnli" else "validation"]
    predict_dataset = raw_datasets["test_matched" if task == "mnli" else "test"]
    # if data_args.max_train_samples is not None:
    #     train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # metric = load_metric("glue", task)
    metric = load_metric("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def warp_data_loader(dataset, batch_size=64):
        data_collator = default_data_collator
        return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=data_collator,
                drop_last=False,
                num_workers=1,
                pin_memory=True,
            )
    train_dataloader = warp_data_loader(train_dataset, batch_size)
    eval_dataloader = warp_data_loader(eval_dataset, batch_size)
    predict_dataloader = warp_data_loader(predict_dataset, batch_size)

    
    acts_inputs = next(iter(train_dataloader))

    del acts_inputs['idx']
    return model, acts_inputs, train_dataloader, eval_dataloader, predict_dataloader, tokenizer


def glue_ft_test(aux_task='cola', tgt_task='sst2'):
    # model1, acts_inputs1, train_dataloader1, eval_dataloader1, predict_dataloader1, tokenizer1 = glue_ft_load(aux_task, tag='bert-base-uncased')  # 80.82 on cola
    model2, acts_inputs2, train_dataloader2, eval_dataloader2, predict_dataloader2, tokenizer2 = glue_ft_load(tgt_task, tag='bert-base-uncased', seed='1')  # 91.74 on sst2
    model1 = copy.deepcopy(model2)
    glue_param_load(model1, aux_task, omit_classifier=True, seed='2')
    
    # acts_inputs = {}
    # for k in acts_inputs1.keys():
    #     acts_inputs[k] = torch.cat([acts_inputs1[k], acts_inputs2[k]], dim=0)

    mlp_args = {
        'model_name': 'bert',
    }
    args = uz.get_args(**mlp_args)

    # activations = None # uz2.compute_activations_across_models_zz(args, [model1, model2], [acts_inputs])

    # activations_gridents = None # uz.compute_activations_gredients_across_models(args, [model1, model2], [acts_inputs])
    # model1.classifier = model2.classifier 

    return args, model1, model2, acts_inputs2, eval_dataloader2, tokenizer2

def prepare_for_same_task(task='mrpc'):
    model1, acts_inputs, train_dataloader1, eval_dataloader1, predict_dataloader1, tokenizer1 = glue_ft_load(task, tag='bert-base-uncased', seed='1')  # 91.74 on sst2
    model2, _, _, _, _, _ = glue_ft_load(task, tag='bert-base-uncased', seed='2') 
    
    mlp_args = {
        'model_name': 'bert',
    }
    args = uz.get_args(**mlp_args)

    # activations = None # uz2.compute_activations_across_models_zz(args, [model1, model2], [acts_inputs])

    # activations_gridents = None # uz.compute_activations_gredients_across_models(args, [model1, model2], [acts_inputs])
    # model1.classifier = model2.classifier 

    return args, model1, model2, acts_inputs, eval_dataloader1, tokenizer1

def save_model(model, task, tag, name):
    save_path = f'/public/data0/NLP/zhangzheng15/transformers/checkpoints/{tag}/saved_models/{task}/{name}'
    torch.save(model.state_dict(), save_path)
    
def transformer_translation_test(dataset, aux_model, tgt_model):
    import os
    print('>> fusion \n  >> aux:{} \n >> tgt:{} '.format(aux_model, tgt_model))
    # dir, model = os.path.split(aux_model)
    # aux_model = TransformerModel.from_pretrained(dir, checkpoint_file=model)
    
    # dir, model = os.path.split(tgt_model)
    # tgt_model = TransformerModel.from_pretrained(dir, checkpoint_file=model)
    aux_model = torch.load(aux_model)
    tgt_model = torch.load(tgt_model)
    
    from fairseq import tasks, options
    from fairseq.dataclass import utils
    from omegaconf import OmegaConf

    aux_model["cfg"]['model'] = utils.convert_namespace_to_omegaconf(aux_model["cfg"]['model'])
    aux_model_cfg = OmegaConf.create(aux_model["cfg"])
    aux_task = tasks.setup_task(aux_model_cfg.task)
    aux_model=aux_task.build_model(aux_model_cfg.model.model)
    
    tgt_model["cfg"]['model'] = utils.convert_namespace_to_omegaconf(tgt_model["cfg"]['model'])
    tgt_model_cfg = OmegaConf.create(tgt_model["cfg"])
    tgt_task = tasks.setup_task(tgt_model_cfg.task)
    tgt_model=tgt_task.build_model(tgt_model_cfg.model.model)

    max_tokens = 16384
    shard_batch_itr = False
    epoch = 1
    disable_iterator_cache=False
    from fairseq import utils
    tgt_task.load_dataset('valid', combine=False, epoch=1)
    seed=0
    batch_iterator_tgt = tgt_task.get_batch_iterator(
            dataset=tgt_task.dataset(tgt_model_cfg.dataset.valid_subset),
            max_tokens=max_tokens,
            max_sentences=tgt_model_cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                tgt_task.max_positions(),
                tgt_model_cfg.model.model.max_source_positions,
                max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=tgt_model_cfg.dataset.required_batch_size_multiple,
            seed=seed,
            num_shards=tgt_model_cfg.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=tgt_model.data_parallel_rank if shard_batch_itr else 0,
            num_workers=0,
            epoch=epoch,
            data_buffer_size=tgt_model_cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )

    aux_task.load_dataset('valid', combine=False, epoch=1)
    batch_iterator_aux = aux_task.get_batch_iterator(
        dataset=aux_task.dataset(aux_model_cfg.dataset.valid_subset),
        max_tokens=max_tokens,
        max_sentences=aux_model_cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            aux_task.max_positions(),
            aux_model_cfg.model.model.max_source_positions,
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=aux_model_cfg.dataset.required_batch_size_multiple,
        seed=seed,
        num_shards=aux_model.data_parallel_world_size if shard_batch_itr else 1,
        shard_id=aux_model.data_parallel_rank if shard_batch_itr else 0,
        num_workers=0,
        epoch=epoch,
        data_buffer_size=aux_model_cfg.dataset.data_buffer_size,
        disable_iterator_cache=disable_iterator_cache,
    )

    model_tgt = tgt_model
    model_aux = aux_model
    tgt_inputs = next(batch_iterator_tgt.next_epoch_itr())['net_input']
    aux_inputs = next(batch_iterator_aux.next_epoch_itr())['net_input']

    mlp_args = {
        'model_name': 'transformer',
    }
    args = uz.get_args(**mlp_args)

    return args, model_aux, model_tgt, aux_inputs, tgt_inputs

def mbart_translation_test(exp='enzh', aux_task='bpe_from_scratch_4', tgt_task='128X'):

    config = AutoConfig.from_pretrained(
        "facebook/bart-base",
    )
    if exp == 'enzh':
        tgt_mbart = MBartModel.from_pretrained(
            "facebook/bart-base",
            ) # trimed_en_XX-zh_CN_128X
        aux_mbart = MBartModel.from_pretrained(
            "facebook/bart-base",
        )
        # en_XX-de_DE_bpe_from_scratch_1_trimed_dict
        for (task, model) in [(aux_task, aux_mbart), (tgt_task, tgt_mbart)]:
            model.task.args.data = '/workspace/data/users/zanchangtong1/data-in/en_XX-zh_CN_small'

    elif exp == 'ende':

        tgt_mbart = MBartModel.from_pretrained(
            "facebook/bart-base",
            config=config
            )
        aux_mbart = MBartModel.from_pretrained(
            "facebook/bart-base",
            config=config
        )
        # checkpoint_file='checkpoint_best.pt',
        # en_XX-de_DE_bpe_from_scratch_1_trimed_dict
        # for (task, model) in [(aux_task, aux_mbart), (tgt_task, tgt_mbart)]:
        #     model.task.args.data = '/workspace/data/users/zanchangtong1/data-in/en_XX-de_DE_small'
        # # assert False
    seed = 222
    max_tokens = 16384
    shard_batch_itr = False
    epoch = 1
    disable_iterator_cache=False
    raw_datasets = load_dataset(
        'wmt16',
        'ro-en',
    )
    train_dataset = raw_datasets["train"]
    eval_datatset = raw_datasets["eval"]
    tgt_mbart.task.load_dataset('valid', combine=False, epoch=1)
    batch_iterator_tgt = tgt_mbart.task.get_batch_iterator(
            dataset=tgt_mbart.task.dataset(tgt_mbart.cfg.dataset.valid_subset),
            max_tokens=max_tokens,
            max_sentences=tgt_mbart.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                tgt_mbart.task.max_positions(),
                tgt_mbart.model.max_positions(),
                max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=tgt_mbart.cfg.dataset.required_batch_size_multiple,
            seed=seed,
            num_shards=tgt_mbart.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=tgt_mbart.data_parallel_rank if shard_batch_itr else 0,
            num_workers=0,
            epoch=epoch,
            data_buffer_size=tgt_mbart.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )

    aux_mbart.task.load_dataset('valid', combine=False, epoch=1)
    batch_iterator_aux = aux_mbart.task.get_batch_iterator(
        dataset=aux_mbart.task.dataset(aux_mbart.cfg.dataset.valid_subset),
        max_tokens=max_tokens,
        max_sentences=aux_mbart.cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            aux_mbart.task.max_positions(),
            aux_mbart.model.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=aux_mbart.cfg.dataset.required_batch_size_multiple,
        seed=seed,
        num_shards=aux_mbart.data_parallel_world_size if shard_batch_itr else 1,
        shard_id=aux_mbart.data_parallel_rank if shard_batch_itr else 0,
        num_workers=0,
        epoch=epoch,
        data_buffer_size=aux_mbart.cfg.dataset.data_buffer_size,
        disable_iterator_cache=disable_iterator_cache,
    )

    model_tgt = tgt_mbart.model
    model_aux = aux_mbart.model
    tgt_inputs = next(batch_iterator_tgt.next_epoch_itr())['net_input']
    aux_inputs = next(batch_iterator_aux.next_epoch_itr())['net_input']

    mlp_args = {
        'model_name': 'mbart',
    }
    args = uz.get_args(**mlp_args)

    # return args, model1, model2, acts_inputs2, eval_dataloader2, tokenizer2
    return args, model_aux, model_tgt, aux_inputs, tgt_inputs



