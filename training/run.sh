#!/bin/bash
GPU=0
# python multi_runs.py --dataset mnist --model VAE --epoch 30 --network large --std 0.4 --noise 0.1
for model in VAE
do
	for dataset in wave
	do
		for network in small large
		do
			echo $model
			echo $dataset
			echo $network
			GPU=$(((GPU+1)%3))
			screen -dmS compare$network$dataset$model bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i multi_runs.py --dataset $dataset --model $model --epochs 75 --network $network --noise 0.1"
		done
	done
done
