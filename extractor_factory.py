from feature_extractor.models.simclr_v2.simclr_v2 import SimCLRv2
from feature_extractor.models.dino.dino import DINO
from feature_extractor.models.plip.plip import PLIP

def build_extractor(checkpoint_path, extractors, version_id, using_extractor='simclr_v2', dataloader=None, custom_weights=False, num_labels=1000):
    
    ext_factory = {
        'simclr_v2': SimCLRv2,
        'dino': DINO,
        'plip': PLIP
    }

    fallback_extractor = 'simclr_v2'

    try:
        versions_dict = extractors[using_extractor].versions
    except KeyError:
        versions_dict = extractors[fallback_extractor].versions

    try:
        if using_extractor == 'dino':
            return ext_factory[using_extractor](dataloader, checkpoint_path, versions_dict, version_id, custom_weights=custom_weights, num_labels=num_labels)
        return ext_factory[using_extractor](dataloader, checkpoint_path, versions_dict)
    except NotImplementedError as e:
        print(f"Error: {e} - Using default fallback extractor - {using_extractor} not implemented.")
        return ext_factory[fallback_extractor](dataloader, checkpoint_path, versions_dict)
