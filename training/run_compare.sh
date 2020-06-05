#!/bin/bash
GPU=0
for model in m1 m2 m3
do
	GPU=$((($GPU+1)%3))
	screen -dmS compare$model bash -c "export CUDA_VISIBLE_DEVICES=1;python -i compare.py --dataset circle --model $model --epochs 200"
done
