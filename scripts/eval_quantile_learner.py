import argparse, torch, ipdb, itertools, sys, os, random, math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import binned_statistic_dd as bin
from torch.utils.data import DataLoader

from distance_network import DistanceNetwork
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
from dataset import DistanceDataset, D4rlDataset
from utils import trajectories_generator

def update_min_distances(X, y, prev_res=None):
    low = [0.48, 0.48, -5.25, -5.25] * 2
    high = [3.7, 3.7, 5.25, 5.25] * 2
    bounds=(low, high)
    nbins = [9, 9, 2, 2] * 2
    custom_min = lambda l: min(l, default=math.inf)

    new_res = bin(X, y, statistic=custom_min, range=list(zip(*bounds)), bins=nbins)
    if prev_res is not None:
        new_res = new_res._replace(statistic=np.minimum(prev_res.statistic, new_res.statistic))
    return new_res

def get_bucket_idxs(X, y):
    low = [0.48, 0.48, -5.25, -5.25] * 2
    high = [3.7, 3.7, 5.25, 5.25] * 2
    bounds=(low, high)
    nbins = [10, 10, 2, 2] * 2
    res = bin(X.numpy(), y, statistic='count', range=list(zip(*bounds)), bins=nbins, expand_binnumbers=True)
    bin_idxs = res.binnumber - 1
    assert np.min(bin_idxs) >= 0, "error with binning"
    assert np.max(bin_idxs) < 9, "error with binning"
    return bin_idxs

def plot_pred_vs_actual(model, get_data, device, savedir, plot_fraction=.001):
    all_pred = []
    all_actual = []
    all_min = []
    res = None

    num_episodes = 500
    # get min distances for each bucket
    for X, predict_y, true_y, img in tqdm(get_data(num_episodes)):
        X = X.numpy()
        res = update_min_distances(X, true_y, res)

    # get min, predicted, actual distances for each data point
    for X, predict_y, true_y, img in tqdm(get_data(num_episodes)):
        if random.random() < plot_fraction:
            bin_idxs = get_bucket_idxs(X, true_y)
            all_min.extend(res.statistic[tuple(bin_idxs)])

            # forward pass
            X = X.to(device).float()
            pred = model(X).squeeze()
            all_pred.extend(pred.detach().cpu().numpy())
            all_actual.extend(true_y.numpy())
    
    # plot predicted vs true and predicted vs min
    plt.scatter(all_actual, all_pred, alpha=0.1, label="actual")
    plt.scatter(all_min, all_pred, alpha=0.1, label="min_distance")
    plt.legend()
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.xlim(-5,175)
    plt.ylim(-5,175)
    plt.title("quantile = 0.05: pred vs actual")
    plt.savefig(f"{savedir}/quantile_pred_vs_actual.png")
    plt.close('all')
    return res


def plot_distances_map(model, device, res, get_data, savedir):
    distances = []
    goal = np.array([3., 1.])
    for X, predict_y, true_y, img in tqdm(D4rlDataset('maze2d-umaze-dense-v1', 500)):
        if np.linalg.norm(goal - X[4:6], ord=2) < 0.25:
            X = torch.tensor(X).to(device)
            pred = model(X)
            bin_idx = get_bucket_idxs(X.cpu().unsqueeze(0), np.array([true_y]))
            min_distance = res.statistic[tuple(bin_idx)][0]
            distances.append((X[0].item(), X[1].item(), true_y, pred.item(), min_distance))
            if len(distances) % 100 == 0:
                print(len(distances))
                if len(distances) == 20000:
                    break


    x, y, true_distance, pred_distance, min_distance = zip(*distances)
    for distance, title in zip([true_distance, pred_distance, min_distance], ['observed', 'predicted', 'min']):
        plt.scatter(x, y, alpha=0.3, c=distance, cmap='viridis')
        plt.xlabel("x pos")
        plt.ylabel("y pos")
        plt.colorbar()
        plt.scatter([goal[0]], [goal[1]], s=70, marker='*', color='gold', zorder=3)
        plt.savefig(f"{savedir}/quantile_map_{title}_distances.png")
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_pred_vs_actual")
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    # set seeding
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(os.path.join(args.model_dir, 'pytorch_model.pt'))
    model = DistanceNetwork(input_dim=8, output_dim=1).to(device)
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device(device))

    # set up dataloader
    data = lambda num_episodes: DataLoader(D4rlDataset('maze2d-umaze-dense-v1', num_episodes), batch_size=512)
    res = plot_pred_vs_actual(model, data, device, args.model_dir)
    plot_distances_map(model, device, res, data, args.model_dir)
