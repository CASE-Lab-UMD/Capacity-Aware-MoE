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

root=/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/electra-base-discriminator/True # TODO electra
#root=/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/bert-base-cased/True # TODO BERT
#root=/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/roberta-base/True # TODO RoBERTa
#root=/user/sunsiqi/hs/MoE/tasks/language-modeling/checkpoints/albert-base-v2/True # TODO Albert

task_name=cola

echo "${root}"
echo "${task_name}"

if [ ! -f ${root}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${root}/log.out
fi

nohup /user/sunsiqi/openfold/lib/conda/envs/adapter/bin/python /user/sunsiqi/hs/MoE/tasks/text-classification/test_loop.py \
      --root ${root} \
      --task_name ${task_name} \
      > ${root}/log.out & echo $! > ${root}/log.txt &