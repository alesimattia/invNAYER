#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --error=err/Train_scratch.err            # standard error file
#SBATCH --output=log/Train_scratch.log           # standard output file
#SBATCH --account=IscrC_MACRO        # account name
#SBATCH --verbose
#SBATCH --cpus-per-task=4
python train_scratch.py --model resnet18 --dataset CIFAR10 --batch-size 512 --lr 0.2 --gpu 0 \
	--epoch 100 --workers 4 --data_root ../ --seed 32