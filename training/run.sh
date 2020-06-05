#!/bin/bash
GPU=0
for model in VAE EM
do
	for dataset in wave circle
	do
		for network in small large
		do
			echo $model
			echo $dataset
			echo $network
			GPU=$(((GPU+1)%3))
			screen -dmS compare$dataset$model bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i multi_runs.py --dataset $dataset --model $model --epochs 100 --network $network --noise 0.02"
		done
	done
done
