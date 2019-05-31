#!/usr/bin/env bash

model_path=/raid/data/kaggle/jigsaw/bert-pretrained-models/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/
output_path=../meta/

python preprocessing/convert_bert_weight.py convert-tf-to-pytorch   --model_path $model_path \
                                                                    --output_path $output_path