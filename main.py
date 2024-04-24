import argparse
import torch
from utils import get_device, load_config
from feature_extractor.models.simclr_v2.simclr_v2 import SimCLRv2
from feature_extractor.models.dino.dino import DINO
from aggregators.models.dsmil.dsmil import DSMIL
from mil import MILNet
from train_mil import train

def get_sample_batch(n, channels=3):
    inputs = []
    for i in range(n):
        inputs.append(torch.randn((channels,255,255)))
    return torch.stack(inputs).to(get_device())

def convalidate_args(args):
    return True

def build_extractor(checkpoint_path, extractors, version_id, using_extractor='simclr_v2', dataloader=None):
    
    ext_factory = {
        'simclr_v2': SimCLRv2,
        'dino': DINO
    }

    fallback_extractor = 'simclr_v2'

    try:
        versions_dict = extractors[using_extractor].versions
    except KeyError:
        versions_dict = extractors[fallback_extractor].versions

    try:
        if using_extractor == 'dino':
            return ext_factory[using_extractor](dataloader, checkpoint_path, versions_dict, version_id)
        return ext_factory[using_extractor](dataloader, checkpoint_path, versions_dict)
    except NotImplementedError as e:
        print(f"Error: {e} - Using default fallback extractor - {using_extractor} not implemented.")
        return ext_factory[fallback_extractor](dataloader, checkpoint_path, versions_dict)

def build_aggregator(aggregator_name, input_size, output_size, custom_weights=False):
    
    aggregator_factory = {
        'dsmil': DSMIL
    }
    
    return aggregator_factory[aggregator_name](input_size, output_size)
            

def main():
    parser = argparse.ArgumentParser(description='Inference DeepMIML method on MIL datasets exploting DiffInfinite')
    parser.add_argument('--datasets', default='musk1', type=str, help='Choose MIL datasets from: TCGA, Camelyon16, Synthetic (diffinfinite)')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epoch', default=40, type=int, help='Number of aggregator total training epochs[40]')
    parser.add_argument('--cv_fold', default=2, type=int, help='Number of cross validation k-fold [10]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='dsmil', type=str, help='Which MIL model [dsmil]')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--config_path', default='config.yml', type=str, help='deepmiml-fw config filepath')

    args = parser.parse_args()
    
    if not convalidate_args(args):
        print('Invalid arguments')
        exit(1)

    
    config = load_config(args.config_path)
    extractor = build_extractor(config.checkpoint_path, config.extractors, config.using_version, config.using_extractor)
    # extractor.print_summary()
    # extractor.benchmark('datasets/ILSVRC2012_img_val/', 5, 32)
    # input_batch = get_sample_batch(3)
    # x = extractor.compute_features(input_batch)
    
    
    aggregator = build_aggregator(config.using_aggregator, extractor.embed_dim, 3, custom_weights=True)
    milnet = MILNet(aggregator, extractor)
    

    # train(milnet, config.data_path, args.num_epoch, args.lr, args.weight_decay, args.batch_size_per_gpu, args.num_workers, args.cv_fold)
    

if(__name__ == '__main__'):
    main()