from aggregators.models.dsmil.dsmil import DSMIL
from aggregators.models.FastMIML.fastmiml import FastMIML
from aggregators.models.BRkNN.brknn import BRkNN
from aggregators.models.CitationKNN.citationknn import CitationKNN
from aggregators.models.MILR.milr import MILR
from aggregators.models.MISVM.mi_svm import MISVM, miSVM
from aggregators.models.MLkNN.MLkNN import MLkNN


def build_aggregator(config, extractor):
    
    aggregator_factory = {
        'dsmil': DSMIL,
        'brknn': BRkNN,
        'cknn': CitationKNN,
        'fastmiml': FastMIML,
        'milr': MILR,
        'miSVM': miSVM,
        'MISVM': MISVM,
        'mlknn': MLkNN
    }
    
    
    if config.using_aggregator not in aggregator_factory:
        raise NotImplementedError(f"Aggregator {config.using_aggregator} not implemented")
    
    if config.using_aggregator == 'dsmil':
        return aggregator_factory[config.using_aggregator](extractor, config.miml_methods[config.using_aggregator])
    
    return aggregator_factory[config.using_aggregator](config.miml_methods[config.using_aggregator])