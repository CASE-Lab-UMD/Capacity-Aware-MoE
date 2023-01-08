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
    pyfile = '/user/sunsiqi/hs/MoE/tasks/text-classification/run_glue.py'
    config_keys = [
                    "--use_moe",
                    "--model_name_or_path",
                    "--output_dir",
                    "--task_name",
                    "--per_device_eval_batch_size",
                    "--resume_from_checkpoint",
                    "--cache_dir",
                    "--overwrite_output_dir",
                    "--do_eval",
                    "--device_ids",
                    "--metric_for_best_model",
                    "--dataloader_num_workers",
                    ]

    for line in open(os.path.join(root, 'config.txt'), 'r').readlines():
        config = line.replace('\n', '')
    config = ['--' + c[:-1] if 'device_ids' not in c else '--' + c.replace('device_ids ', 'device_ids "')[:-1] + '"' for c in config.split('--')[1:]]
    config = [' ' + c for c in config if c.split(' ')[0] in config_keys]
    # config[0] = config[0][1:]
    for i in range(len(config)):
        if 'cache_dir' in config[i]:
            config[i] = ' --cache_dir ' + './cache_dir/' + task_name
    config.append(' --task_name ' + task_name)

    for sub_dir in tqdm(os.listdir(root)):
        if sub_dir.startswith('checkpoint'):
            dir_name = os.path.join(root, sub_dir)
            state_file = os.path.join(dir_name, 'trainer_state.json')
            for i in range(len(config)):
                if '--output_dir' in config[i]:
                    config[i] = ' --output_dir ' + dir_name + '/' + task_name
            if os.path.exists(dir_name + '/' + task_name + '/' + sub_dir) and len(os.listdir(dir_name + '/' + task_name + '/' + sub_dir)) != 0:
                continue
            with open(state_file, 'r') as f:
                train_state = json.load(f)
            log_history = train_state['log_history']
            # todo find out the pretrained loss
            last_eval = log_history[-1]
            eval_loss = last_eval['eval_loss']
            resume_from_ckpt = dir_name
            config.append(" --resume_from_checkpoint " + resume_from_ckpt)
            cmd = '/user/sunsiqi/openfold/lib/conda/envs/adapter/bin/python ' + pyfile + ''.join(config)
            # todo the argument of test
            os.system(cmd)

if __name__ == "__main__":
    main()