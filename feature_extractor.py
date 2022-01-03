from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, state):
        pass

class NoopFeatureExtractor(FeatureExtractor):
    def extract_features(self, state):
        return state
