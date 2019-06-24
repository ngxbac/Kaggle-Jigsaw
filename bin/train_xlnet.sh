#!/usr/bin/env bash

seed=9374
depth=16
maxlen=220
batch_size=16
accumulation_steps=8
model_name=xlnet

CUDA_VISIBLE_DEVICES=1,2,3 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps \
                                                        --model_name=$model_name