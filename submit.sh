#!/bin/bash

#SBATCH -p hawkcpu,engc,enge,eng
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 9
#SBATCH -J con_sindy

module load anaconda3

conda activate sindy

$SLURM_JOB_CPUS_PER_TASK

########################################################################################################################
# Kosir system and LS problem
# python main.py --system kosir --noise_study "0, 0.1, 0.2" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.05, 0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1, 1" &
# python main.py --system kosir --experiments_study "2, 4, 6" --max_workers 9 --problem LS --dt 0.05 \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.05, 0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1, 1" &
# python main.py --system kosir --sampling_study "0.01, 0.05, 0.1" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.05, 0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1, 1" &


# menten system and LS problem
# python main.py --system menten --noise_study "0, 0.1, 0.2" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.01, 0.1, 1" --stlsq_alpha "0, 0.01, 0.1" &
# python main.py --system menten --experiments_study "2, 4, 6" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 6 --dt 0.01 --stlsq_threshold "0.05, 0.1, 1" --stlsq_alpha "0, 0.01, 0.1" &
# python main.py --system menten --sampling_study "0.01, 0.05, 0.1" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.01, 0.1, 1" --stlsq_alpha "0, 0.01, 0.1" &


# Carb system and LS problem
# python main.py --system carb --noise_study "0, 0.1, 0.2" --max_workers 9 --problem LS --seed 20 \
# --degree "1, 2" --nexpt 25 --stlsq_threshold "0.01, 0.09, 0.1, 1, 10" --stlsq_alpha "0.1, 1, 10, 20, 50, 100, 200" &
# python main.py --system carb --experiments_study "15, 20, 25" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 25 --stlsq_threshold "0.01, 0.09, 0.1, 1, 10" --stlsq_alpha "0.1, 1, 10, 20, 100" &
# python main.py --system carb --sampling_study "0.01, 0.05, 0.1" --max_workers 9 --problem LS \
# --degree "1, 2" --nexpt 25 --stlsq_threshold "0.01, 0.09, 0.1, 1, 10" --stlsq_alpha "0.1, 1, 10, 20" &


########################################################################################################################
# Kosir system and NLS problem
# python main.py --system kosir --noise_study "0, 0.1, 0.2" --max_workers 9 --problem NLS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.05, 0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1, 1, 10" &
# python main.py --system kosir --experiments_study "2, 4, 6" --max_workers 9 --problem NLS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.05, 0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1, 1, 10" &
# python main.py --system kosir --sampling_study "0.01, 0.05, 0.1" --max_workers 9 --problem NLS \
# --degree "1, 2" --nexpt 6 --stlsq_threshold "0.05, 0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1, 1, 10" &



########################################################################################################################
# stiffness study 
# python main.py --stiffness_study 1 --max_workers 9 --dt 0.01 &
python main.py --stiffness_study 1 --max_workers 9 --dt 0.001


########################################################################################################################
# mechanism study 
# python main.py --mechanism_study 1 --max_workers 9 --stlsq_threshold "0.1, 0.5, 1" --stlsq_alpha "0, 0.01, 0.1" --degree "1, 2" &


wait
exit
