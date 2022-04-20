#! /bin/sh

#SBATCH --job-name=prop_first_train
#SBATCH --output=prop_first_train.out
#SBATCH --error=prop_first_train.err
#SBATCH --partition=studentbatch
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

srun python -m span_identification --config configs/si_config.yml --do_train --do_eval --model_name_or_path roberta-large --use_crf --overwrite_output_dir