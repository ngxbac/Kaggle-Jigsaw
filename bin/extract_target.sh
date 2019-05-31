#!/usr/bin/env bash

csv_file=/raid/data/kaggle/jigsaw/train.csv
output_path=../meta/
dataset=train

python preprocessing/extract_data.py train-target      --csv_file $csv_file \
                                                       --output_path $output_path