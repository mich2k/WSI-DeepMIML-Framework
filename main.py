import argparse
from extractor_factory import build_extractor
from miml_factory import build_method
from utils import load_config
from mil import MILNet

def convalidate_args(args):

    if args.config_path == '':
        return False

    if args.test_extractor not in [True, False]:
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Inference DeepMIML method on MIL datasets exploting DiffInfinite')
    parser.add_argument('--test_extractor', default=False, type=bool, help='Test the model')
    parser.add_argument('--config_path', default='config.yml', type=str, help='deepmiml-fw config filepath')

    args = parser.parse_args()
    
    if not convalidate_args(args):
        print('Invalid arguments')
        exit(1)

    config = load_config(args.config_path)
    extractor = build_extractor(config, custom_weights=True)
    
    if args.test_extractor:    
        extractor.print_summary()
        extractor.benchmark('datasets/ILSVRC2012_img_val/', 5, 32)
            
    miml_method = build_method(config, extractor)
    milnet = MILNet(miml_method, extractor)
    milnet.train_model()
    
if __name__ == '__main__':
    main()