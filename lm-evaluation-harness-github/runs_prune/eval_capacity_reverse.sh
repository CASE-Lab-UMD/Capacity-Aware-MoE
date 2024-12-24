#!/usr/bin/bash
##############################################################################

export CUDA_VISIBLE_DEVICES=6

export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

expert_capacity=1.0

# strategy=last
# strategy=first
strategy=random
# strategy=score

##############################################################################

autoawq=False
# autoawq=True


# num_epochs=1
# granularity=attn_sequence
# router_only=True



############################# pretrained #############################
# pretrained=/workspace/models/Meta-Llama-3-8B-Instruct
pretrained=/workspace/models/deepseek-moe-16b-base-temp
pretrained=/workspace/models/OLMoE-1B-7B-temp


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


# task=gsm8k
# # rm -rfv $router_dir/$task.json

# nohup lm_eval \
#   --model hf \
#   --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
#   --tasks $task \
#   --num_fewshot 5 \
#   --batch_size auto \
#   --output_path $output_path/$task.json \
#   > $output_path/${task}.out 2>&1 &
#   # --limit 50 \

# wait
# exit 0

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

task=openbookqa

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
