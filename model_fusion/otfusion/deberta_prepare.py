import sys
import torch
ROOT='/workspace/zhangzheng/otfusion'
sys.path.append(ROOT)
import otfusion.utils_zz as uz
import otfusion.utils_zz2 as uz2
import transformers
from torch import nn
import copy

DEBERTA_CLZS = {
    "deberta-base": {
        "config": transformers.DebertaConfig, 
        "model": transformers.DebertaForMaskedLM,
        "tokenizer": transformers.DebertaTokenizer
    },
    "deberta-v2-xxlarge": {
        "config": transformers.DebertaV2Config,
        "model": transformers.DebertaV2Model,
        "tokenizer": transformers.DebertaV2Tokenizer
    }
}

def get_input_ids(tokenizer, batch_num=64):
    text_file = "/workspace/zhangzheng/datasets/formatted_one_article_per_line/wikicorpus_en_one_article_per_line.txt"
    lines = open(text_file).readlines()
    max_seq_len = 128
    batch_input_ids = []
    num = 0
    for line in lines:
        if num >= batch_num:
            break 
        line_tokens = tokenizer.tokenize(line)
        for i in range(0, len(line_tokens)-max_seq_len+2, max_seq_len-2):
            tokens = ['[CLS]'] + line_tokens[i: i+max_seq_len-2] + ['[SEP]']
            batch_input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
            num += 1
    return torch.tensor(batch_input_ids, dtype=torch.int)


def load_head(classifier, pretrained_model_path):
    print("loading parameters of classifier")
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    
    checkpoint_keys = ['lm_predictions.lm_head.dense.bias', 'lm_predictions.lm_head.dense.weight', 
        'lm_predictions.lm_head.LayerNorm.weight', 'lm_predictions.lm_head.LayerNorm.bias', 
        'deberta.embeddings.word_embeddings.weight', 'lm_predictions.lm_head.bias']
    model_keys = ['cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 
        'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 
        'cls.predictions.decoder.weight', 'cls.predictions.bias']
    model_keys = [s.lstrip("cls.") for s in model_keys]
    print(pretrained_state_dict.keys)

    pretrained_cls_state = {}
    for old_key, new_key in zip(checkpoint_keys, model_keys):
        pretrained_cls_state[new_key] = pretrained_state_dict[old_key]
    for name, p in classifier.named_parameters():
        print(name, p.shape)
    print("-------")
    for name, p in pretrained_cls_state.items():
        print(name, p.shape)
    classifier.load_state_dict(pretrained_cls_state, strict=False)


def glue_ft_load(task, tag='deberta-base', max_seq_length=128, batch_size=256, seed='1'):
    from DeBERTa.deberta import load_vocab, tokenizers
    from DeBERTa.apps.tasks import get_task
    from DeBERTa import apps
    from DeBERTa.data import SequentialSampler, BatchSampler
    import parser
    parser = apps.run.build_argument_parser()
    # parser.parse_known_args()
    TASK = {"cola":"CoLA"}.get(task, task.upper())
    vocab_type = 'gpt2'
    args = parser.parse_args(f"--model_config config.json --tag base --task_name {TASK} --data_dir /workspace/zhangzheng/others/DeBERTa/datasets/glue/glue_tasks/{TASK} \
            --init_model /workspace/zhangzheng/others/DeBERTa/checkpoints/{tag}/saved_models/{task}/pytorch_model{seed}/pytorch_model.bin \
            --vocab_path /workspace/zhangzheng/others/DeBERTa/datasets/pretrained/{tag}/bpe_encoder.bin \
            --model_config /workspace/zhangzheng/others/DeBERTa/datasets/pretrained/{tag}/config.json \
            --vocab_type {vocab_type} --output_dir /workspace/zhangzheng/others/DeBERTa/checkpoints/{tag}/saved_models/{task}/debug \
            --max_seq_len 128".split()) # --num_train_epochs 4 --warmup 500 --learning_rate 2e-5 --train_batch_size 32 --cls_drop_out 0.15 --do_train 

    vocab_path, vocab_type = apps.run.load_vocab(args.vocab_path, vocab_type=args.vocab_type, pretrained_id=args.init_model)
    tokenizer = tokenizers[vocab_type](vocab_path)
    task = get_task(args.task_name)(tokenizer = tokenizer, args=args, max_seq_len = args.max_seq_length, data_dir = args.data_dir)
    label_list = task.get_labels()

    eval_data = task.eval_data(max_seq_len=args.max_seq_length)
    if args.do_predict:
        test_data = task.test_data(max_seq_len=args.max_seq_length)

    if args.do_train:
        train_data = task.train_data(max_seq_len=args.max_seq_length, debug=args.debug)
    else:
        train_data = None
    model_class_fn = task.get_model_class_fn()
    model = apps.run.create_model(args, len(label_list), model_class_fn)
    
    def warp_data_loader(dataset, batch_size=256, batch_sampler=None):
        return torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler, 
                num_workers=2,
                pin_memory=True,
            )
    # train_dataloader = warp_data_loader(train_dataset, batch_size)
    assert len(eval_data) == 1
    eval_sampler = SequentialSampler(len(eval_data[0].data))
    batch_sampler = BatchSampler(eval_sampler, batch_size)
    eval_dataloader = warp_data_loader(eval_data[0].data, batch_size, batch_sampler=batch_sampler)
    # predict_dataloader = warp_data_loader(predict_dataset, batch_size)

    
    acts_inputs = next(iter(eval_dataloader))
    predict_loader = None
    return model, acts_inputs, train_data, eval_dataloader, predict_loader, tokenizer


def glue_param_load(model: nn.Module, task, tag='bert-base-uncased', model_name="pytorch_model", omit_classifier=True, seed='1'):
    # model_name_or_path = f'/public/data0/NLP/zhangzheng15/transformers/checkpoints/{tag}/saved_models/{task}/{model_name}{seed}/pytorch_model.bin'
    model_name_or_path = f'/workspace/zhangzheng/others/DeBERTa/checkpoints/{tag}/saved_models/{task}/pytorch_model{seed}/pytorch_model.bin'
    
    checkpoint = torch.load(model_name_or_path, map_location=torch.device('cpu'))
    if omit_classifier:
        omit_layers = [layer for layer in checkpoint.keys() if "classifier" in layer]
        for layer in omit_layers:
            del checkpoint[layer]
    model.load_state_dict(checkpoint, strict=False)
    
def glue_ft_test(aux_task='cola', tgt_task='sst2', tag='deberta-base'):
    # model1, acts_inputs1, train_dataloader1, eval_dataloader1, predict_dataloader1, tokenizer1 = glue_ft_load(aux_task, tag='bert-base-uncased')  # 80.82 on cola
    model2, acts_inputs2, train_dataloader2, eval_dataloader2, predict_dataloader2, tokenizer2 = glue_ft_load(tgt_task, tag=tag, seed='1')  # 91.74 on sst2
    model1 = copy.deepcopy(model2)
    glue_param_load(model1, aux_task, omit_classifier=True, seed='2', tag=tag)
    
    # acts_inputs = {}
    # for k in acts_inputs1.keys():
    #     acts_inputs[k] = torch.cat([acts_inputs1[k], acts_inputs2[k]], dim=0)

    mlp_args = {
        'model_name': tag,
    }
    args = uz.get_args(**mlp_args)

    # activations = None # uz2.compute_activations_across_models_zz(args, [model1, model2], [acts_inputs])

    # activations_gridents = None # uz.compute_activations_gredients_across_models(args, [model1, model2], [acts_inputs])
    # model1.classifier = model2.classifier 

    return args, model1, model2, acts_inputs2, eval_dataloader2, tokenizer2


def prepare_for_same_task(task='mrpc', tag='deberta-base'):
    model1, acts_inputs, train_dataloader1, eval_dataloader1, predict_dataloader1, tokenizer1 = glue_ft_load(task, tag=tag, seed='1')  # 91.74 on sst2
    model2, _, _, _, _, _ = glue_ft_load(task, tag=tag, seed='2') 
    
 
    mlp_args = {
        'model_name': 'bert',
    }
    args = uz.get_args(**mlp_args)

    # activations = None # uz2.compute_activations_across_models_zz(args, [model1, model2], [acts_inputs])

    # activations_gridents = None # uz.compute_activations_gredients_across_models(args, [model1, model2], [acts_inputs])
    # model1.classifier = model2.classifier 

    return args, model1, model2, acts_inputs, eval_dataloader1, tokenizer1

def deberta_test(tag="deberta-base"):
    import utils_zz2 as uz2
    import transformers
    mlp_args = {
        'model_name': 'deberta_v2',
    }
    args = uz.get_args(**mlp_args)

    # MODEL_PATH = "/public/data0/NLP/zhangzheng15/datasets/pretrained/deberta-v2-xxlarge"
    # MODEL_PATH2 = "/public/data0/NLP/zhangzheng15/datasets/pretrained/deberta-v2-xxlarge-mnli"
    MODEL_PATH = f"/public/data0/NLP/users/zhangzheng15/datasets/pretrained/{tag}"
    MODEL_PATH2 = f"/public/data0/NLP/users/zhangzheng15/datasets/pretrained/{tag}-mnli"
    
    # DebertaV2Model, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering, DebertaV2ForSequenceClassification, DebertaV2PreTrainedModel
    model_config = DEBERTA_CLZS[tag]['config'].from_pretrained(MODEL_PATH)
    hf_model = DEBERTA_CLZS[tag]['model'].from_pretrained(MODEL_PATH, config=model_config)
    load_head(hf_model.cls, f"{MODEL_PATH}/pytorch_model.bin")
    tokenizer = DEBERTA_CLZS[tag]['tokenizer'].from_pretrained(MODEL_PATH, config=model_config)

    model_config2 = DEBERTA_CLZS[tag]['config'].from_pretrained(MODEL_PATH2)
    hf_model2 = DEBERTA_CLZS[tag]['model'].from_pretrained(MODEL_PATH2, config=model_config2)
    load_head(hf_model2.cls, f"{MODEL_PATH}/pytorch_model.bin")
    tokenizer2 = DEBERTA_CLZS[tag]['tokenizer'].from_pretrained(MODEL_PATH2, config=model_config2)

    # sys.path.append('/workspace/zhangzheng/bing_bert')
    # import v2.model_load as ml
    # data_args = ml.construct_arguments()
    # batch, dict_input, dataset_iterator = ml.load_batch(data_args, 64)
    # sys.path.remove('/workspace/zhangzheng/bing_bert')
    input_ids = get_input_ids(tokenizer)
    acts_inputs = [{'input_ids': input_ids}]
    # activations = None
    activations = uz2.compute_activations_across_models_zz(args, [hf_model, hf_model2], acts_inputs)

    return args, (hf_model, hf_model2), (model_config, model_config2), (tokenizer, tokenizer2), activations, acts_inputs
    # pretrain_dataset_provider.prefetch_shard(index + 1)


def save_model(model, task, tag, name):
    save_path = f'/public/data0/NLP/users/zhangzheng15/others/DeBERTa/checkpoints/{tag}/saved_models/{task}/{name}'
    torch.save(model.state_dict(), save_path)