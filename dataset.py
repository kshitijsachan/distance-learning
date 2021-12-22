import torch, random, ipdb, pickle
import numpy as np
from more_itertools import chunked, flatten, windowed
from collections import defaultdict

from utils import parse_example

class DistanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, generate_trajs, feature_extractor, label_mapping_func):
        super(DistanceDataset).__init__()
        self.generate_trajs = generate_trajs
        self.transform = feature_extractor.extract_features
        self.label_mapping_func = label_mapping_func
        self.num_episodes_to_group = 5

        # DELETE THIS
        with open('true_distance.pkl', 'rb') as f:
            self.true_distance = pickle.load(f)
        self.true_distance = defaultdict(lambda: random.randint(0, 500), self.true_distance)
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
                to_predict_y = self.label_mapping_func(true_y)
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
                    # DELETE THIS
                    k = parse_example(x)
                    if k not in self.true_distance:
                        ipdb.set_trace()
                        self.cntr += 1
                        print(self.cntr)

                    to_predict_y = self.label_mapping_func(self.true_distance[k])
                    yield (x, to_predict_y, true_y, img)

class TripletLossDataset(torch.utils.data.IterableDataset):
    def __init__(self, generate_trajs, feature_extractor, idle_threshold=100, positive_radius=5):
        super(DistanceDataset).__init__()
        self.generate_trajs = generate_trajs
        self.transform = feature_extractor.extract_features
        self.idle_threshold = idle_threshold
        self.positive_radius = positive_radius
    
    def __iter__(self):
        for traj in self.generate_trajs():
            n = len(traj)
            ram_traj, img_traj = self.transform(traj)
            ram_traj = np.array(ram_traj).astype(np.float32)
            anchor_idxs = list(range(self.positive_radius, n - self.positive_radius))
            random.shuffle(anchor_idxs)
            for anchor_idx in anchor_idxs:
                lower_lim, upper_lim = anchor_idx - self.positive_radius, anchor_idx + self.positive_radius
                positive_idxs = list(range(max(0, lower_lim), min(n, upper_lim))) 
                random.shuffle(positive_idxs)
                for positive_idx in positive_idxs:
                    for _ in range(10):
                        negative_idx = random.randint(0, n - 1)
                        if lower_lim - self.idle_threshold <= negative_idx <= upper_lim + self.idle_threshold:
                            break
                    else:
                        continue

                    anchor, positive, negative = ram_traj[anchor_idx], ram_traj[positive_idx], ram_traj[negative_idx]
                    anchor_img, positive_img, negative_img = img_traj[anchor_idx], img_traj[positive_idx], img_traj[negative_idx]
                    yield (anchor, positive, negative, (anchor_img, positive_img, negative_img))

