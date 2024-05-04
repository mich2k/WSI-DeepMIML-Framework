from methods.MIML.dsmil.dsmil import DSMIL
from methods.MIML.FastMIML.fastmiml import FastMIML
from methods.ML.BRkNN.brknn import BRkNN
from methods.ML.MLkNN.MLkNN import MLkNN
from methods.MI.CitationKNN.citationknn import CitationKNN
from methods.MI.MILR.milr import MILR
from methods.MI.MISVM.mi_svm import MISVM


def build_method(config, extractor):
    
    method_factory = {
        'dsmil': DSMIL,
        'brknn': BRkNN,
        'cknn': CitationKNN,
        'fastmiml': FastMIML,
        'milr': MILR,
        'MISVM': MISVM,
        'mlknn': MLkNN
    }
    
    
    if config.method not in method_factory:
        raise NotImplementedError(f"Methodology {config.method} not implemented")
    
    # !!
    if config.method == 'dsmil':
        return method_factory[config.method](extractor, config.miml_methods[config.method])
    if config.method == 'misvm':
        return method_factory[config.method](config.miml_methods[config.method], config.miml_methods[config.method].C, config.miml_methods[config.method].kernel)

    
    return method_factory[config.method](config.miml_methods[config.method])