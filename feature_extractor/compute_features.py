import feature_extractor.models.simclrv1 as simclrv1


class FeatureExtractor():    
    def __init__(self, model):
        if model.extractor == 'simclr_v1':
            self.extractor = simclrv1()
        else:
            print('Extractor not implemented')
            exit(1)