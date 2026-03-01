#!/bin/sh
n_epochs=90
batch_size=256
model_architecture="VisionTransformer"
dataset="ImageNet"
model_save_dir="/home/yslcoat/trained_models"
data_dir="/home/yslcoat/data/imagenet1k/"

systemd-inhibit --what=sleep python ../train.py \
    --dataset "$dataset" \
    --imagenet-root "$data_dir" \
    --arch "$model_architecture" \
    --epochs "$n_epochs" \
    --batch-size "$batch_size" \
