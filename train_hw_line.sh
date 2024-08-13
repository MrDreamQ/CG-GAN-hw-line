#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
getdate=$(date +%Y%m%d_%H%M%S)

nohup python train_hw_line.py > ./logs/train_hw_line.${getdate}.log 2>&1 &
