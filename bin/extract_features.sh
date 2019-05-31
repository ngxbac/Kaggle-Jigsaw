#!/usr/bin/env bash

csv_file=/raid/data/kaggle/jigsaw/train.csv
output_path=../meta/
dataset=train

python preprocessing/extract_data.py meta-features      --csv_file $csv_file \
                                                        --dataset $dataset \
                                                        --output_path $output_path