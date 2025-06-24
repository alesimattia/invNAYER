#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --error=err/Train_scratch_new.err            # standard error file
#SBATCH --output=log/Train_scratch_new.log           # standard output file
#SBATCH --account=IscrC_MACRO        # account name
#SBATCH --verbose
#SBATCH --cpus-per-task=4
python train_scratch.py --workers 4 --gpu 0 --batch-size 512 --lr 0.2  \
	--epochs 100  --data_root ../ --model resnet18 --dataset cifar10  --seed 32 \
	--footprint