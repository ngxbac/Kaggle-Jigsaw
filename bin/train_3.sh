#!/usr/bin/env bash

seed=7328
depth=12
maxlen=220
batch_size=32
accumulation_steps=4

CUDA_VISIBLE_DEVICES=2 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps