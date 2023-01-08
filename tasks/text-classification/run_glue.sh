#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --array=0-4%5

# wandb env variables
# model list
# small: klue/roberta-small, aajrami/bert-mlm-small
# base: bert-base-cased, roberta-base
#todo model_name_or_path=/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/klue/roberta-small # roberta-small
#model_name_or_path=/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/aajrami/bert-mlm-small # bert-small
model_name_or_path=roberta-base
#todo model_name_or_path=aajrami/bert-mlm-small
metric_for_best_model=accuracy # accuracy for mrpc、rte、sst2、qnli、mnli、qqp
TASK_NAME=qnli

#todo metric_for_best_model=matthews_correlation # matthews_correlation for cola
#TASK_NAME=cola

#todo metric_for_best_model=pearson # pearson for stsb
#TASK_NAME=stsb

do_train=False
per_device_train_batch_size=24
per_device_eval_batch_size=24
cache_dir=./cache_dir/${TASK_NAME}
num_train_epochs=10
dataloader_num_workers=16
save_strategy=epoch
evaluation_strategy=epoch
save_steps=500
eval_steps=500
weight_decay=0.1
warmup_ratio=0.06
learning_rate=1e-5
seed=42
device_ids="4 5 6 7"
use_moe=True
log_out=log.out
output_dir=./checkpoints/${TASK_NAME}/${model_name_or_path##*/}
resume_from_checkpoint="/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/klue/roberta-small/copy"

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --use_moe ${use_moe} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --resume_from_checkpoint ${resume_from_checkpoint} \
      --cache_dir ${cache_dir} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --warmup_ratio ${warmup_ratio} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval \
      --device_ids "${device_ids}" \
      --eval_steps ${eval_steps} \
      --save_steps ${save_steps} \
      --seed ${seed} --metric_for_best_model ${metric_for_best_model} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      > ${output_dir}/config.txt

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

nohup /user/sunsiqi/openfold/lib/conda/envs/adapter/bin/python /user/sunsiqi/hs/MoE/tasks/text-classification/run_glue.py \
      --use_moe ${use_moe} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --resume_from_checkpoint ${resume_from_checkpoint} \
      --cache_dir ${cache_dir} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval \
      --device_ids "${device_ids}" \
      --eval_steps ${eval_steps} \
      --save_steps ${save_steps} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --warmup_ratio ${warmup_ratio} \
      --seed ${seed} --metric_for_best_model ${metric_for_best_model} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      > ${output_dir}/${log_out} & echo $! > ${output_dir}/log.txt &