#!/usr/bin/bash
##############################################################################
# version=3
# size=8

# mod_n=20
# data_type=mixed
# data_type=evol_instruct
# gradient_scale=0.0
# learning_rate=1e-4
# max_train_samples=1000
# weight_decay=0.1

export CUDA_VISIBLE_DEVICES=1

export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

expert_capacity=1.0

# strategy=last
# strategy=first
# strategy=random
strategy=score

# folder_name="llama${version}-${size}b-mod"
# folder_name="llama${version}-${size}b-instruct-mod"
# folder_name="qwen-2.5-7b-mod"
# folder_name="mistral-7b-mod"

# folder_name="Qwen-2.5-1.5B"
# folder_name="Qwen-2.5-14B"


# pretrained=/workspace/MoD/results/ckpt/${folder_name}
# pretrained=/workspace/models/Qwen2.5-1.5B
# pretrained=/workspace/models/Qwen2.5-14B
# pretrained=/workspace/MoD/results/ckpt/qwen-2.5-14b-mod

# export CUDA_VISIBLE_DEVICES=1
# folder_name="attn_sequence_epoch1_lr2e-4_mod_n16_gradient_scale0.0_wd0."
# pretrained=/workspace/MoD/trained_models/mixed/True/${folder_name}
# pretrained=/workspace/MoD/results/ckpt/${folder_name}
# pretrained=/workspace/models/Qwen2.5-7B
# pretrained=/workspace/models/Qwen2.5-3B
# pretrained=/workspace/models/Mistral-7B-v0.3


##############################################################################

autoawq=False
# autoawq=True

# num_epochs=1
# granularity=attn_sequence
# router_only=True

############################# pretrained #############################
# pretrained=/workspace/models/Meta-Llama-3-8B-Instruct
pretrained=/workspace/models/deepseek-moe-16b-base-temp
# pretrained=/workspace/models/OLMoE-1B-7B-temp


# root_path="/workspace/MoD/"
# router_dir=$pretrained
# output_path=$router_dir
output_path=$pretrained/expert_capacity-$expert_capacity/$strategy

# rm -r $output_path
mkdir -p $output_path

echo $pretrained


if [ ! -d "$pretrained" ]; then
  exit 0
fi

# if [ ! -d "$router_dir" ]; then
#   exit 0
# fi

# mkdir -p $router_dir/$folder_name

# echo $pretrained

task=openbookqa

# nohup 
lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
#   > $output_path/${task}.out 2>&1 &

exit 0
wait


task=piqa
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

# exit 0
wait


task=rte
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

# exit 0
wait


task=winogrande
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

# exit 0 
wait


task=boolq
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=arc_challenge
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 25 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=hellaswag
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 10 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=mmlu
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=gsm8k
# rm -rfv $router_dir/$task.json

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &
  # --limit 50 \

wait
# exit 0