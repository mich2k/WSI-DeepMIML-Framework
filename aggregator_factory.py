from aggregators.models.dsmil.dsmil import DSMIL
from aggregators.models.FastMIML.fastmiml import FastMIML
from aggregators.models.BRkNN.brknn import BRkNN
from aggregators.models.CitationKNN.citationknn import CitationKNN
from aggregators.models.MILR.milr import MILR
from aggregators.models.MISVM.mi_svm import MISVM, miSVM
from aggregators.models.MLkNN.MLkNN import MLkNN


def build_aggregator(aggregator_name, input_size, output_size, use_pytorch=False):
    
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
    
    
    if aggregator_name not in aggregator_factory:
        raise NotImplementedError(f"Aggregator {aggregator_name} not implemented")
    
    if not use_pytorch:
        return aggregator_factory[aggregator_name]()
    
    return aggregator_factory[aggregator_name](input_size, output_size)