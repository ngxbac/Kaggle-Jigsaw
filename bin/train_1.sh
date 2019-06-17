#!/usr/bin/env bash

seed=23759
depth=11
maxlen=300
batch_size=64
accumulation_steps=2
model_name=bert

CUDA_VISIBLE_DEVICES=2,3 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps \
                                                        --model_name=$model_name