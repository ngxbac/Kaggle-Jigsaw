#!/usr/bin/env bash

seed=7782
depth=12
maxlen=300
batch_size=32
accumulation_steps=4

CUDA_VISIBLE_DEVICES=3 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps