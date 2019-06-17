#!/usr/bin/env bash

model_path=/raid/bac/kaggle/jigsaw/gpt2_weight/
csv_file=/raid/data/kaggle/jigsaw/test.csv
output_path=../features_300_gpt/
dataset=test
max_sequence_length=300

python preprocessing/extract_data.py extract-data-gpt2   --model_path $model_path \
                                                    --csv_file $csv_file \
                                                    --dataset $dataset \
                                                    --output_path $output_path \
                                                    --max_sequence_length $max_sequence_length