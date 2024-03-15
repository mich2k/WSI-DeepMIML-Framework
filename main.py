import os, glob
import argparse




def main():
    parser = argparse.ArgumentParser(description='Inference DeepMIML method on MIL datasets')
    parser.add_argument('--datasets', default='musk1', type=str, help='Choose MIL datasets from: musk1, musk2, elephant, fox, tiger [musk1]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epoch', default=40, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--cv_fold', default=2, type=int, help='Number of cross validation fold [10]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='dsmil', type=str, help='Which MIL model [dsmil]')
    args = parser.parse_args()
    