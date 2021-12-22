import torch, random, ipdb, pickle
import numpy as np
from more_itertools import chunked, flatten
from collections import defaultdict

from utils import parse_example

class DistanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, generate_trajs, feature_extractor):
        super(DistanceDataset).__init__()
        self.generate_trajs = generate_trajs
        self.transform = feature_extractor.extract_features
        self.num_episodes_to_group = 5

    def traj_to_data_pairs(self, traj):
        ram_traj, img_traj = self.transform(traj)
        n = len(traj)
        data = []
        for start_idx in range(n):
            for end_idx in range(start_idx, n):
                start_state = ram_traj[start_idx]
                end_state = ram_traj[end_idx]
                x = np.concatenate((start_state, end_state))
                true_y = end_idx - start_idx
                to_predict_y = true_y
                start_img = img_traj[start_idx]
                end_img = img_traj[end_idx]
                image_pair = (start_img, end_img)
                data.append((x, to_predict_y, true_y, image_pair))
        return data

    def shuffle_data(self, data):
        """each element of data is the list of all pairs from a single trajectory"""
        flat_data = list(flatten(data))
        random.shuffle(flat_data)
        return flat_data
    
    def __iter__(self):
        while True:
            data_generator = map(self.traj_to_data_pairs, self.generate_trajs())
            # shuffle pairs between and across episodes
            shuffled_data_generator = map(self.shuffle_data, chunked(data_generator, self.num_episodes_to_group))
            for trajs in shuffled_data_generator:
                for x, to_predict_y, true_y, img in trajs:
                    k = parse_example(x)
                    yield (x, to_predict_y, true_y, img)

