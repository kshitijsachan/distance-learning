import torch, itertools, math, pickle
from tqdm import tqdm

from collections import defaultdict
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
from dataset import DistanceDataset
from utils import trajectories_generator, parse_example

if __name__ == '__main__':
    feature_extractor = MontezumaRamFeatureExtractor()
    train_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_train.pkl.gz"), feature_extractor, lambda x: x)
    test_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"), feature_extractor, lambda x: x)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128)

    min_distances = defaultdict(lambda: 9999999999999999)
    train_episodes = 100
    train_size = int(300 * 601 * train_episodes / 128)
    for X, predict_y, true_y, img in tqdm(itertools.islice(train_dataloader, 0, train_size), total=train_size):
        for x, y in zip(X, true_y):
            key = parse_example(x)
            min_distances[key] = min(min_distances[key], y.item())

    test_episodes = 30
    test_size = int(300 * 601 * test_episodes / 128)
    for X, predict_y, true_y, img in tqdm(itertools.islice(test_dataloader, 0, test_size), total=test_size):
        for x, y in zip(X, true_y):
            key = parse_example(x)
            min_distances[key] = min(min_distances[key], y.item())

    with open('true_distance.pkl', 'wb') as f:
        pickle.dump(dict(min_distances), f)

