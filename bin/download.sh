#!/usr/bin/env bash

for i in 1 2 3 4 5 6 7 8 9 a b c d e f; do
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_${i}.tar.gz train_${i}.tar.gz
done