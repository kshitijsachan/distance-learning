import argparse, torch, ipdb, itertools, sys, os, random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from distance_network import DistanceNetwork
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
from dataset import DistanceDataset
from utils import trajectories_generator


def plot_pred_vs_actual(model, data, device, plot_fraction=.01):
    all_pred = []
    all_actual = []

    num_episodes = 100
    num_examples = int(num_episodes * (550 * 551 / 2))
    for X, predict_y, true_y, img in tqdm(itertools.islice(data, 0, num_examples), total=num_examples):
        if random.random() < plot_fraction:
            X = torch.tensor(X).to(device).float()
            pred = model(X).squeeze()
            all_pred.append(pred.item())
            all_actual.append(true_y)
    
    # plot
    plt.scatter(all_pred, all_actual, alpha=0.1)
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("quantile = 0.05: pred vs actual")
    plt.savefig("run_data/quantile_pred_vs_actual.png")
    plt.close('all')
    ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_pred_vs_actual")
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    # set seeding
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path)
    model = DistanceNetwork(input_dim=12, output_dim=1).to(device)
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device(device))

    # set up dataloader
    feature_extractor = MontezumaRamFeatureExtractor()
    data_path = "/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"
    data = DistanceDataset(lambda: trajectories_generator(data_path), feature_extractor)

    plot_pred_vs_actual(model, data, device)
