#! /bin/sh

#SBATCH --job-name=prop_eval
#SBATCH --output=prop_eval.out
#SBATCH --error=prop_eval.err
#SBATCH --partition=studentbatch
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

srun python -m span_identification --config configs/si_config.yml --do_eval --use_crf
