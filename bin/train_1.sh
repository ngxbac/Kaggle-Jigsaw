#!/usr/bin/env bash

seed=6037
depth=11
maxlen=220
batch_size=32
accumulation_steps=4

CUDA_VISIBLE_DEVICES=0 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps