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
        extractor_conf = config.extractors[config.using_extractor]
        versions_dict = extractor_conf.versions
        return ext_factory[config.using_extractor](config.trainset_path, config.checkpoint_path, versions_dict, extractor_conf)

    except (NotImplementedError, KeyError) as e:
        print(f"Error: {e} - Using default fallback extractor - {config.using_extractor} not implemented.")
        extractor_conf = config.extractors[fallback_extractor]
        return ext_factory[fallback_extractor](config.trainset_path, config.checkpoint_path, extractor_conf.versions, extractor_conf)
