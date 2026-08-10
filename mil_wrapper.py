from methods.baseline import Baseline
from feature_extractor.extractor.extractor import FeatureExtractor
from utils import DiffInfiniteDataset
class MILWrapper():
    def __init__(self, name: str, method: Baseline, extractor: FeatureExtractor):
        super().__init__()
        self.name = name
        self.model = method
        self.extractor = extractor
    
    def train_model(self, train_path: str, test_path: str):

        if 'diffinfinite' not in train_path or 'diffinfinite' not in test_path:
            raise NotImplementedError(f"Unsupported dataset paths: {train_path}, {test_path}")

        trainset = DiffInfiniteDataset(train_path, stop_at=800)
        testset = DiffInfiniteDataset(test_path, stop_at=800)

        #if model is OG_DSMIL or MLkNN we need to pass the extractor
        if self.name == 'og_dsmil' or self.name == 'mlknn':
            self.model.run(trainset, testset, self.extractor)
        else:
            self.model.run(trainset, testset)
        
        
        