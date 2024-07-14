import argparse
from extractor_factory import build_extractor
from miml_factory import build_method
from utils import load_config
from mil_wrapper import MILWrapper


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
    
    miml_method = build_method(config, extractor)
    milnet = MILWrapper(config.method, miml_method, extractor)
    milnet.train_model(config.trainset_path, config.testset_path)
    
if __name__ == '__main__':
    main()