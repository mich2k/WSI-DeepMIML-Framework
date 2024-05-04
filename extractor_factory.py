from feature_extractor.models.simclr_v2.simclr_v2 import SimCLRv2
from feature_extractor.models.dino.dino import DINO
from feature_extractor.models.plip.plip import PLIP

def build_extractor(config, dataloader=None, custom_weights=False, num_labels=1000):
    
    ext_factory = {
        'simclr_v2': SimCLRv2,
        'dino': DINO,
        'plip': PLIP
    }

    fallback_extractor = 'simclr_v2'

    try:
        versions_dict = config.extractors[config.using_extractor].versions
    except KeyError:
        versions_dict = config.extractors[fallback_extractor].versions

    try:
        extractor_conf = config.extractors[config.using_extractor]
        return ext_factory[config.using_extractor](config.data_path, config.checkpoint_path, versions_dict, extractor_conf)

    except NotImplementedError as e:
        print(f"Error: {e} - Using default fallback extractor - {config.using_extractor} not implemented.")
        return ext_factory[fallback_extractor](dataloader, config.checkpoint_path, versions_dict)
