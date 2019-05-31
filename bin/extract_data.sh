#!/usr/bin/env bash

model_path=/raid/data/kaggle/jigsaw/bert-pretrained-models/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/
csv_file=/raid/data/kaggle/jigsaw/train.csv
output_path=../meta/
dataset=train
max_sequence_length=220

python preprocessing/extract_data.py extract-data   --model_path $model_path \
                                                    --csv_file $csv_file \
                                                    --dataset $dataset \
                                                    --output_path $output_path \
                                                    --max_sequence_length $max_sequence_length