#!/usr/bin/env bash

seed=6037
depth=24
maxlen=300
batch_size=32
accumulation_steps=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps