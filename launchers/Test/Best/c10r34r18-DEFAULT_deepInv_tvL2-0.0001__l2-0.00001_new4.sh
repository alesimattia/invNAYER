#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --error=err/ERR_c10r34r18-DEFAULT_deepInv_tvL2-0.0001__l2-0.00001_new4.err            # standard error file
#SBATCH --output=log/LOG_c10r34r18-DEFAULT_deepInv__tvL2-0.0001__l2-0.00001_new4.log           # standard output file
#SBATCH --account=IscrC_MACRO       # account name
#SBATCH --verbose
#SBATCH --cpus-per-task=4
python3 datafree_kd.py --workers 4 --gpu 0 --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --warmup 20 \
 --epochs 120 --dataset cifar10 --method nayer --lr_g 4e-3 --teacher resnet34 --student resnet18 \
 --save_dir run/ --adv 1.33 --bn 10.0 --oh 0.5 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
 --contr 0 --log_tag c10r34r18-DEFAULT_deepInv_tvL2-0.0001__l2-0.00001_new4 \
 --l2 0.00001 --tv_l2 0.0001 \
 --footprint --metrics "Off" --r_feature 0.01