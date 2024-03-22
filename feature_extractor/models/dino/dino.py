from feature_extractor.extractor.extractor import FeatureExtractor


class DINO(FeatureExtractor):
    def __init__(self, version="dino_resnet50"):
        super().__init__()
            
        self.model = None

    
    def load_weights(self, apply_mlp=False):
        return super().load_weights(apply_mlp)
    
    def compute_features(self):
        return super().compute_features()