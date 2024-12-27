#!/usr/bin/bash
##############################################################################

export CUDA_VISIBLE_DEVICES=1

expert_capacity=1.0
  
# strategy=last
# strategy=first
# strategy=random
strategy=score

##############################################################################

autoawq=False
# autoawq=True

############################# pretrained #############################
pretrained=./models/deepseek-moe-16b-base-temp
# pretrained=./models/OLMoE-1B-7B-temp

output_path=$pretrained/expert_capacity-$expert_capacity/$strategy

mkdir -p $output_path

echo $pretrained


if [ ! -d "$pretrained" ]; then
  exit 0
fi


task=openbookqa

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

exit 0
wait


task=piqa

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=rte

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=winogrande

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=boolq

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

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,router_dir=$router_dir,expert_capacity=$expert_capacity,strategy=$strategy,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait
