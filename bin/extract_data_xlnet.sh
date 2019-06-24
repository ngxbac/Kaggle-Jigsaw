#!/usr/bin/env bash

model_path=/raid/data/kaggle/jigsaw/xlnet/xlnet_cased_L-24_H-1024_A-16/
csv_file=/raid/data/kaggle/jigsaw/train.csv
output_path=../features_220_xlnet/
dataset=train
max_sequence_length=220

python preprocessing/extract_data.py extract-data-xlnet   --model_path $model_path \
                                                    --csv_file $csv_file \
                                                    --dataset $dataset \
                                                    --output_path $output_path \
                                                    --max_sequence_length $max_sequence_length