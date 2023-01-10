import os
import json
import torch
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--root", type=str, default='/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/albert-base-v2/True',
        help="the root of pretrained models."
    )
    parser.add_argument(
        "--task_name", type=str, default='cola', help="the task to evaluate."
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    root = args.root
    task_name = args.task_name
    dictionary = {}
    file_list = os.listdir(root)
    file_list.sort()
    # todo gather zero-shot performance
    # for sub_dir in tqdm(file_list):
    #     if sub_dir.startswith('checkpoint'):
    #         try:
    #             dir_name = root + '/' + sub_dir + '/' + task_name + '/' + sub_dir
    #             eval_file = os.path.join(dir_name, 'eval_results.json')
    #             with open(eval_file, 'r') as f:
    #                 eval_results = json.load(f)
    #             eval_loss = eval_results['eval_loss']
    #             dictionary[int(sub_dir.split('-')[-1])] = eval_loss
    #         except:
    #             continue
    # todo gather pretrained performance
    trainer_state = os.path.join(root, 'trainer_state.json')
    with open(trainer_state, 'r') as f:
        trainer_state = json.load(f)
    f.close()
    log_history = trainer_state['log_history']
    steps = [elem['step'] for elem in log_history if 'eval_loss' in elem]
    losses = [elem['eval_loss'] for elem in log_history if 'eval_loss' in elem]
    dictionary = dict(zip(steps, losses))
    dictionary = sorted(dictionary.items(), key=lambda kv:kv[0])
    print(dictionary)

if __name__ == "__main__":
    main()