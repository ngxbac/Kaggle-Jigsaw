#!/usr/bin/env bash

data_dir=/raid/data/kaggle/jigsaw/
output_path=../lstm_features/
crawl_embedding_path=/raid/data/kaggle/jigsaw/embedding/crawl-300d-2M.pkl
glove_embedding_path=/raid/data/kaggle/jigsaw/embedding/glove.840B.300d.pkl

python preprocessing/extract_data_lstm.py extract-data   --data_dir $data_dir \
                                                         --output_path $output_path \
                                                         --crawl_embedding_path $crawl_embedding_path \
                                                         --glove_embedding_path $glove_embedding_path