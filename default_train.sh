#!/bin/sh
n_epochs=90
batch_size=256
model_architecture="FeedForwardNeuralNetwork"
dataset="DummyDataset"
model_save_dir="/home/yslcoat/trained_models"

systemd-inhibit --what=sleep python train.py \
    --data "$dataset" \
    --arch "$model_architecture" \
    --epochs "$n_epochs" \
    --batch-size "$batch_size" \
