import torch, random, ipdb
import numpy as np

class DistanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, generate_trajs, feature_extractor):
        super(DistanceDataset).__init__()
        self.generate_trajs = generate_trajs
        self.transform = feature_extractor.extract_features

        # discount factor
        self.GAMMA = 0.99

    def _discount_distance(self, distance):
        return (1 - self.GAMMA ** distance) / (1 - self.GAMMA)
    
    def __iter__(self):
        while True:
            for _episode_num, traj in enumerate(self.generate_trajs()):
                # extract features from trajectory
                traj = self.transform(traj)
                n = len(traj)
                # Randomly shuffle start and end state
                start_idxs = list(range(n))
                random.shuffle(start_idxs)
                for start_idx in start_idxs:
                    end_idxs = list(range(start_idx, n))
                    random.shuffle(end_idxs)
                    for end_idx in end_idxs:
                        start_state = traj[start_idx]
                        end_state = traj[end_idx]
                        yield (
                                np.concatenate((start_state, end_state)),
                                self._discount_distance(end_idx - start_idx)
                                )
