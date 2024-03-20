#!/bin/bash

python3 verify.py -pth_path r101_1x_sk0.pth -val_path ILSVRC2012_img_val/ -n_samples 2 -batch_size 32
