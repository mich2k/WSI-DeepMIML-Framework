import argparse
from extractor_factory import build_extractor
from aggregator_factory import build_aggregator
from utils import load_config
from mil import MILNet

def convalidate_args(args):
    if args.num_epoch <= 0:
        return False
    if args.cv_fold <= 0:
        return False
    if args.weight_decay < 0:
        return False
    if args.num_workers < 0:
        return False
    if args.config_path == '':
        return False
    if args.num_classes <= 0:
        return False
    if args.test_extractor not in [True, False]:
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Inference DeepMIML method on MIL datasets exploting DiffInfinite')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epoch', default=40, type=int, help='Number of aggregator total training epochs[40]')
    parser.add_argument('--cv_fold', default=2, type=int, help='Number of cross validation k-fold [10]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset')
    parser.add_argument('--test_extractor', default=False, type=bool, help='Test the model')
    parser.add_argument('--config_path', default='config.yml', type=str, help='deepmiml-fw config filepath')

    args = parser.parse_args()
    
    if not convalidate_args(args):
        print('Invalid arguments')
        exit(1)

    config = load_config(args.config_path)
    extractor = build_extractor(config.checkpoint_path, config.extractors, config.using_version, config.using_extractor, custom_weights=True)
    
    if args.test_extractor:    
        extractor.print_summary()
        extractor.benchmark('datasets/ILSVRC2012_img_val/', 5, 32)
        
    aggregator = build_aggregator(config.using_aggregator, extractor.embed_dim, args.num_classes)
    milnet = MILNet(aggregator, extractor)
    milnet.train_model()
    
if __name__ == '__main__':
    main()