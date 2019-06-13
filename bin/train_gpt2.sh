#!/usr/bin/env bash

seed=6037
depth=12
maxlen=220
batch_size=64
accumulation_steps=2
model_name=gpt2

CUDA_VISIBLE_DEVICES=0,1 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps \
                                                        --model_name=$model_name