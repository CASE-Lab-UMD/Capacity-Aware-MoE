import torch
import os
import json
from transformers import AutoModel, AutoTokenizer

# state_dict = torch.load("/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/roberta-small/True/pytorch_model.bin", map_location='cpu')
# todo roberta-base
# roberta_base_model = AutoModel.from_pretrained("roberta-base")
# roberta_base_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# bert_model = AutoModel.from_pretrained("aajrami/bert-mlm-small")
# bert_tokenizer = AutoTokenizer.from_pretrained("aajrami/bert-mlm-small")
# print()
# roberta_model = AutoModel.from_pretrained("Unbabel/xlm-roberta-comet-small")
# roberta_tokenizer = AutoTokenizer.from_pretrained("Unbabel/xlm-roberta-comet-small")
# print()
with open('/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/xlm-roberta-comet-small/pretrained/xlm-roberta-comet-small/', 'r') as f:
    protein_dict = json.load(f)

# # todo summation
# def summation():
#     for key in state_dict:
#         # todo exclude attention layers
#         if 'attention' not in key:
#             for elem in RESET_DICT:
#                 if elem in key:
#                     new_key = key.split(elem)[0] + key.split(elem)[-1].replace(key.split(elem)[-1].split('.')[0] + '.', '')
#                     TEMP[new_key] += state_dict[key]
#                     Redundant.append(key)
# def average():
#     times = {}
#     for key in state_dict:
#         # todo exclude attention layers
#         if 'attention' not in key:
#             for elem in RESET_DICT:
#                 if elem in key:
#                     new_key = key.split(elem)[0] + key.split(elem)[-1].replace(key.split(elem)[-1].split('.')[0] + '.', '')
#                     TEMP[new_key] += state_dict[key]
#                     if new_key not in times:
#                         times[new_key] = 1
#                     else:
#                         times[new_key] += 1
#                     Redundant.append(key)
#     for new_key in TEMP.keys():
#         TEMP[new_key] = TEMP[new_key] / times[new_key]
#
# method = 'copy'
# reload_methods = {'avg': average, 'sum': summation}
# # todo deal with roberta
# MODEL = 'pytorch_model.bin'
# STATE_DIR = '/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/klue/roberta-small'
# file = os.path.join(STATE_DIR, MODEL)
# state_dict = torch.load(file, map_location="cpu")
# POS_KEY = 'output.dense'
# RESET_DICT = {'dense.experts.': ''}
# NEW_KEY = {}
# TEMP = {}
# Redundant = []
# if method != "copy":
#     # todo detect the new keys in the fused model.
#     for key in state_dict:
#         # todo exclude attention layers
#         if 'attention' not in key:
#             for elem in RESET_DICT:
#                 if elem in key:
#                     new_key = key.split(elem)[0] + key.split(elem)[-1].replace(key.split(elem)[-1].split('.')[0] + '.', '')
#                     if new_key not in NEW_KEY:
#                         NEW_KEY[new_key] = key
#     # todo zero out all elements in output.dense.weight/bias.
#     for new_key in NEW_KEY:
#         TEMP[new_key] = torch.zeros(size=state_dict[NEW_KEY[new_key]].size())
#
#     # todo reset the elements in output.dense.weight/bias based on pretrained experts
#     func = reload_methods[method]
#     func()
#     for new_key in NEW_KEY:
#         state_dict[new_key] = TEMP[new_key]
#     # todo remove the redundant keys in experts
#     for key in Redundant:
#         del state_dict[key]
#     if not os.path.exists(file.replace(MODEL, method)):
#         os.mkdir(file.replace(MODEL, method))
#     torch.save(state_dict, file.replace(MODEL, method + '/' + MODEL))
# else:
#     if not os.path.exists(file.replace(MODEL, method)):
#         os.mkdir(file.replace(MODEL, method))
#     torch.save(state_dict, file.replace(MODEL, method + '/' + MODEL))
#
