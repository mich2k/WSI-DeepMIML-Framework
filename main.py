import os, glob
import argparse

from feature_extractor.models.simclr_v2.simclr_v2 import SimCLRv2

def convalidate_args(args):
    return True

def build_extractor(model='simclr_v2'):
    
    extractors = {
        'simclr_v2': SimCLRv2
    }
    
    if model not in extractors:
        raise NotImplementedError(f'Not implemented extractor: {model}')
    
    return extractors[model]()


def main():
    parser = argparse.ArgumentParser(description='Inference DeepMIML method on MIL datasets')
    parser.add_argument('--datasets', default='musk1', type=str, help='Choose MIL datasets from: TCGA, Camelyon16, Synthetic (diffinfinite)')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epoch', default=40, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--cv_fold', default=2, type=int, help='Number of cross validation ma poi è fold [10]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--extractor', default='simclr_v2', type=str, help='Which MIL model [simclrv2]')
    parser.add_argument('--model', default='dsmil', type=str, help='Which MIL model [dsmil]')
    args = parser.parse_args()
    
    if not convalidate_args(args):
        print('Invalid arguments')
        exit(1)
    
    extractor = build_extractor('simclr_v2')
    extractor.print_summary()

if(__name__ == '__main__'):
    main()