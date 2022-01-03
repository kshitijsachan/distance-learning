import torch, random, ipdb, pickle, gym, d4rl
import numpy as np
from itertools import islice
from more_itertools import chunked, flatten
from collections import defaultdict

from torch.utils.data import IterableDataset
from utils import parse_example

class DistanceDataset(IterableDataset):
    def __init__(self, trajs, feature_extractor, num_episodes):
        super(DistanceDataset).__init__()
        self.trajs = trajs 
        self.transform = feature_extractor.extract_features
        self.num_episodes = num_episodes

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

    @staticmethod
    def scramble(gen, buffer_size=100000):
        buf = []
        i = iter(gen)
        while True:
            try:
                e = next(i)
                buf.append(e)
                if len(buf) >= buffer_size:
                    choice = random.randint(0, len(buf)-1)
                    buf[-1], buf[choice] = buf[choice], buf[-1]
                    yield buf.pop()
            except StopIteration:
                random.shuffle(buf)
                yield from buf
                return
    
    def __iter__(self):
        # shuffle pairs between and across episodes
        data_generator = self.scramble(flatten(map(self.traj_to_data_pairs, islice(self.trajs, self.num_episodes))))
        for x, to_predict_y, true_y, img in data_generator:
            yield (x, to_predict_y, true_y, img)

class D4rlDataset(IterableDataset):
    def __init__(self, mdp_name, num_episodes):
        super(D4rlDataset).__init__()
        env = gym.make(mdp_name)
        self.trajs = d4rl.sequence_dataset(env)
        self.num_episodes = num_episodes

    def traj_to_data_pairs(self, traj):
        obs = traj['observations']
        n = len(obs)
        data = []
        for start_idx in range(n):
            for end_idx in range(start_idx, n):
                start_state = obs[start_idx]
                end_state = obs[end_idx]
                x = np.concatenate((start_state, end_state))
                true_y = end_idx - start_idx
                to_predict_y = true_y
                data.append((x, to_predict_y, true_y))
        return data

    @staticmethod
    def scramble(gen, buffer_size=100000):
        buf = []
        i = iter(gen)
        while True:
            try:
                e = next(i)
                buf.append(e)
                if len(buf) >= buffer_size:
                    choice = random.randint(0, len(buf)-1)
                    buf[-1], buf[choice] = buf[choice], buf[-1]
                    yield buf.pop()
            except StopIteration:
                random.shuffle(buf)
                yield from buf
                return
    
    def __iter__(self):
        # shuffle pairs between and across episodes
        data_generator = self.scramble(flatten(map(self.traj_to_data_pairs, islice(self.trajs, self.num_episodes))))
        for x, to_predict_y, true_y in data_generator:
            yield (x, to_predict_y, true_y, 0) # add 0 at the end because there is no image
