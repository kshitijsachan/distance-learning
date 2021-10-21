import torch, random
import numpy as np

class DistanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, generate_trajs, feature_extractor):
        super(DistanceDataset).__init__()
        self.generate_trajs = generate_trajs
        self.transform = feature_extractor.extract_features
    
    def __iter__(self):
        while True:
            for traj in self.generate_trajs():
                n = len(traj)
                # Randomly shuffle start and end state
                start_idxs = list(range(n))
                random.shuffle(start_idxs)
                for i in start_idxs:
                    end_idxs = list(range(i, n))
                    random.shuffle(end_idxs)
                    for j in end_idxs:
                        start_state, _start_image = traj[i]
                        end_state, _end_image = traj[j]
                        start_features = self.transform(start_state)
                        end_features = self.transform(end_state)
                        yield np.concatenate((start_features, end_features)), j - i, (_start_image, _end_image)
