#!/bin/bash
GPU=0
for width in 6
do
	for depth in 1
	do
		for leakiness in 0.0 0.1 -1.0
		do
			for scale in 1 2 3
			do
				GPU=$((($GPU+1)%7))
	       			screen -dmS compare$width$depth$leakiness$scale bash -c "export CUDA_VISIBLE_DEVICES='';python compare.py --dataset circle --width $width --depth $depth --leakiness $leakiness --scale $scale"
			done
		done
	done
done
