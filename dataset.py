import torch, random
import numpy as np

class DistanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, generate_trajs, feature_extractor):
        super(DistanceDataset).__init__()
        self.generate_trajs = generate_trajs
        self.transform = feature_extractor.extract_features
    
    def __iter__(self):
        while True:
            for episode_num, traj in enumerate(self.generate_trajs()):
                n = len(traj)
                # Randomly shuffle start and end state
                start_idxs = list(range(n))
                random.shuffle(start_idxs)
                for start_idx in start_idxs:
                    end_idxs = list(range(start_idx, n))
                    random.shuffle(end_idxs)
                    for end_idx in end_idxs:
                        start_state, _start_image = traj[start_idx]
                        end_state, _end_image = traj[end_idx]
                        start_features = self.transform(start_state)
                        end_features = self.transform(end_state)
                        yield (
                                np.concatenate((start_features, end_features)), 
                                end_idx - start_idx, 
                                (_start_image, _end_image),
                                episode_num
                                )
