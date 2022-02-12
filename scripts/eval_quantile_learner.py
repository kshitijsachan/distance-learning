import argparse, torch, ipdb, itertools, sys, os, random, math, gym, d4rl
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import binned_statistic_dd as bin
from torch.utils.data import DataLoader

from distance_network import DistanceNetwork
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
from dataset import EvalD4rlDataset
from utils import trajectories_generator, envs_dict


class EvalQuantileLearner():
    def __init__(self, model_dir, seed=0):
        # set seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # set up model
        self.mdp_name = self.get_mdp_name(model_dir)
        self.env = gym.make(self.mdp_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(os.path.join(model_dir, 'pytorch_model.pt'))
        self.model = DistanceNetwork(input_dim=10, output_dim=1).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(torch.device(self.device))

        goal_dict = {
                'maze2d-umaze-dense-v1' : [3., 0.],
                'maze2d-medium-v1' : [6., 5.],
                'maze2d-large-v1' : [5., 4.],
                'antmaze-umaze-v0' : [3., 0.],
                }
        self.goal = np.array(goal_dict[self.mdp_name])
        self.threshold = 0.08
        self.model_dir = model_dir

        # plot distances
        self.plot_distances_map()

    def get_mdp_name(self, model_dir):
        with open(os.path.join(model_dir, 'run_command.txt')) as f:
            command = f.readlines()[0]
        mdp_name = command.split(' ')[1]
        return envs_dict[mdp_name]

    def plot_distances_map(self):
        pred_states = []
        pred_distances = []
        observed_states = []
        observed_distances = []
        for traj in tqdm(EvalD4rlDataset(self.mdp_name)):
            traj = np.array(traj)
            goals = np.broadcast_to(self.goal, (traj.shape[0], self.goal.shape[0]))
            # randomize actuator velocities
            targets = np.array([self.env.observation_space.sample()[2:] for _ in range(len(traj))])
            X = np.concatenate((traj, goals, targets), axis=1)
            X = torch.tensor(X).float().to(self.device)
            pred = self.model(X).squeeze(axis=1).cpu().tolist()
            X = X.cpu().numpy()

            for i in range(len(traj)):
                if random.random() < .5:
                    pred_states.append(X[i, :2])
                    pred_distances.append(pred[i])

            for i, s in enumerate(traj):
                if np.linalg.norm(self.goal - s[:2], ord=2) < self.threshold:
                    for j in range(i + 1):
                        observed_states.append(traj[j, :2])
                        observed_distances.append(i - j)
                    break

        for state, distance, title in zip([observed_states, pred_states], [observed_distances, pred_distances], ['observed', 'predicted']):
            if state:
                x, y = zip(*state)
                plt.scatter(x, y, alpha=0.8, c=distance, cmap='viridis')
                plt.xlabel("x pos")
                plt.ylabel("y pos")
                plt.colorbar()
                plt.scatter([self.goal[0]], [self.goal[1]], s=90, marker='*', color='gold', zorder=3)
                plt.savefig(f"{self.model_dir}/quantile_map_{title}_distances.png")
                plt.close('all')

goal_dict = {
        'antmaze-umaze-v0' : [0.75, 8.75],
        'maze2d-umaze-dense-v1' : [1, 1], 
        'maze2d-umaze-v1' : [1, 1], 
        'maze2d-medium-v1' : [6, 6],
        'maze2d-large-v1' : [7, 9]
        }

def plot_distances_map(mdp_name, model, device, savedir):
    # get full goal state
    env_name = envs_dict[mdp_name]
    env = gym.make(env_name)
    goal = np.array(goal_dict[env_name])
    full_goal_state = None
    closest_distance = math.inf
    for traj in tqdm(EvalD4rlDataset(env_name)):
        traj = np.array(traj)
        for s in traj:
            distance = np.linalg.norm(goal - s[:2], ord=2)
            if distance < closest_distance:
                closest_distance = distance
                full_goal_state = s
    full_goal_state[:2] = goal
    goal = full_goal_state

    # store predicted and observed distances
    pred_states = []
    pred_distances = []
    observed_states = []
    observed_distances = []
    threshold = 0.08
    for traj in tqdm(EvalD4rlDataset(env_name)):
        traj = np.array(traj)
        goals = np.broadcast_to(goal, (traj.shape[0], goal.shape[0]))
        X = np.concatenate((traj, goals), axis=1)
        X = torch.tensor(X).float().to(device)
        pred = model(X).squeeze(axis=1).cpu().tolist()
        X = X.cpu().numpy()

        for i in range(len(traj)):
            if random.random() < .05:
                pred_states.append(X[i, :2])
                pred_distances.append(pred[i])

        for i, s in enumerate(traj):
            if np.linalg.norm(goal[:2] - s[:2], ord=2) < threshold:
                for j in range(i + 1):
                    observed_states.append(traj[j, :2])
                    dis = i - j
                    observed_distances.append(dis)
                break

    for state, distance, title in zip([observed_states, pred_states], [observed_distances, pred_distances], ['observed', 'predicted']):
        if state:
            x, y = zip(*state)
            plt.scatter(x, y, alpha=0.8, c=distance, cmap='viridis')
            plt.xlabel("x pos")
            plt.ylabel("y pos")
            plt.colorbar()
            plt.clim(0, 500)
            plt.scatter([goal[0]], [goal[1]], s=90, marker='*', color='gold', zorder=3)
            plt.savefig(f"{savedir}/quantile_map_{title}_distances.png", bbox_inches='tight', pad_inches=0)
            plt.close('all')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_pred_vs_actual")
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    #res = plot_pred_vs_actual(model, data, device, args.model_dir)
    EvalQuantileLearner(args.model_dir)
