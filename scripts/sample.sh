#! /usr/bin/bash


# Set memory and GPU requirements for the job
#$ -l tmem=20G
#$ -l gpu=true
#$ -pe gpu 2
#$ -N ljiang_ddpm_sample
#$ -o /SAN/medic/MRpcr/logs/med_ddpm_output.log
#$ -e /SAN/medic/MRpcr/logs/med_ddpm_error.log
#$ -wd /SAN/medic/MRpcr
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate med

python3 /SAN/medic/MRpcr/med-ddpm/sample.py --inputfolder /SAN/medic/MRpcr/med-ddpm/dataset/whole_head/mask --exportfolder /SAN/medic/MRpcr/med-ddpm/exports/ --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --batchsize 1 --num_samples 1 --weightfile /SAN/medic/MRpcr/med-ddpm/model/model_128.pt
