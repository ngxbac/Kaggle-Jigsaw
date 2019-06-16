#!/usr/bin/env bash

declare -i seed=2106
for fold in 0 1 2 3 4; do
    seed=$seed+$fold
    depth=11
    maxlen=300
    batch_size=128
    accumulation_steps=1

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main_kfold.py train    --seed=$seed \
                                                            --depth=$depth \
                                                            --maxlen=$maxlen \
                                                            --batch_size=$batch_size \
                                                            --accumulation_steps=$accumulation_steps \
                                                            --fold=$fold
done