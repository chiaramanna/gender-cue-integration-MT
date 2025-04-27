#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --job-name=translation       
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate env_t

python scripts/attribute.py \
    --input_file "$1" \
    --translation_file "$2" \
    --model_name "$3" \
    --src_lang "$4" \
    --tgt_lang "$5" \
    --alignment_file "$6" \
    --suffix "$7" 