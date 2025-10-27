#!/usr/bin/bash
##############################################################################

export CUDA_VISIBLE_DEVICES=1

##############################################################################

autoawq=False

############################# pretrained #############################
pretrained=./models/deepseek-moe-16b-base-temp
pretrained=./models/deepseek-moe-16b-base
pretrained=./models/OLMoE-1B-7B-0924

output_path=$pretrained

echo $pretrained


if [ ! -d "$pretrained" ]; then
  exit 0
fi



batch_size=auto

task=openbookqa

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=piqa

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=rte

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=winogrande

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=boolq

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 0 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=arc_challenge

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 25 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=hellaswag

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 10 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=mmlu

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait


task=gsm8k

nohup lm_eval \
  --model hf \
  --model_args pretrained=$pretrained,parallelize=True,trust_remote_code=True,dtype="bfloat16",autoawq=$autoawq \
  --tasks $task \
  --num_fewshot 5 \
  --batch_size $batch_size \
  --output_path $output_path/$task.json \
  > $output_path/${task}.out 2>&1 &

wait
