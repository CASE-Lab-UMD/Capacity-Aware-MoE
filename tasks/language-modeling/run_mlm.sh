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
#model_name_or_path=klue/roberta-small # from korean language understanding not small enough
# todo roberta-base-moe: 3500, bert-base-cased-moe: 3500
#model_name_or_path=Unbabel/xlm-roberta-comet-small # todo roberta-small
#model_name_or_path=aajrami/bert-mlm-small # todo bert-small
#model_name_or_path=bert-base-cased # todo bert-base
#model_name_or_path=roberta-base # todo roberta-base
model_name_or_path=albert-base-v2 # todo albert
#model_name_or_path=google/electra-base-discriminator # todo electra
dataset_name=wikitext
dataset_config_name=wikitext-2-raw-v1
per_device_train_batch_size=12
per_device_eval_batch_size=12
cache_dir=./cache_dir/${dataset_name}_${dataset_config_name}
num_train_epochs=150 # todo 150 for electra, 50 for bert and roberta
dataloader_num_workers=16
save_strategy=steps
evaluation_strategy=steps
save_steps=500
eval_steps=500
seed=42
device_ids="0 1"
#device_ids="0 1 2 3"
use_moe=True
log_out=log.out
output_dir=./checkpoints/${model_name_or_path##*/}/${use_moe}
echo "${output_dir}"
mkdir -p ${output_dir}

echo  --use_moe ${use_moe} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --cache_dir ${cache_dir} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train \
      --do_eval \
      --device_ids "${device_ids}" \
      --eval_steps ${eval_steps} \
      --save_steps ${save_steps} \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      > ${output_dir}/config.txt

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

nohup /user/sunsiqi/openfold/lib/conda/envs/adapter/bin/python /user/sunsiqi/hs/MoE/tasks/language-modeling/run_mlm.py \
      --use_moe ${use_moe} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --cache_dir ${cache_dir} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train \
      --do_eval \
      --device_ids "${device_ids}" \
      --eval_steps ${eval_steps} \
      --save_steps ${save_steps} \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      > ${output_dir}/${log_out} & echo $! > ${output_dir}/log.txt &