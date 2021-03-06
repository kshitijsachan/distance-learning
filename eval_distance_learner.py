import argparse, torch, ipdb, itertools, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor 
from distance_network import DistanceNetwork
from dataset import DistanceDataset
<<<<<<< HEAD
from utils import trajectories_generator
from distance_learner import quantile_loss

def parse_example(example):
    example = example.tolist()
    size = int(len(example) / 2)
    s1, s2 = example[:size], example[size:]
    return _parse_state(s1), _parse_state(s2)

def _parse_state(state):
    xy = _parse_xy(state)
    lives = int(state[6])
    has_key = bool(state[14])
    # left_door_open = bool(state[7])
    # right_door_open = bool(state[8])
    # return (xy, lives, has_key, left_door_open, right_door_open)
    return (xy, lives, has_key)

def _parse_xy(state):
    if state[15]:
        return 'rope'

    x, y = state[0], state[1]
    if state[16]:
        if x < 36:
            return 'left-ladder'
        if x > 119:
            return 'right-ladder'
        return 'middle-ladder'

    if y >= 235:
        if x < 54:
            return 'top-left'
        if 63 < x < 89:
            return 'top-middle'
        if x > 98:
            return 'top-right'

    if y <= 165:
        return 'bottom'

    if 129 <= y <= 209:
        if x < 36:
            return 'middle-left'
        if 56 < x < 101:
            return 'middle-middle'
        if x > 119:
            return 'middle-right'
    return 'misc'

def get_all_combos():
        xy_pos = ['left-ladder', 'middle-ladder', 'right-ladder', 'rope', 'top-left', 'top-middle', 'top-right', 'bottom', 'middle-left', 'middle-middle', 'middle-right', 'misc']
        lives_left = list(range(6))
        has_key = [True, False]
        all_combos = lambda : itertools.product(xy_pos, lives_left, has_key)
        return all_combos
=======
from utils import trajectories_generator, parse_example, get_all_combos
from distance_learner import bucket_distance3 as bucket_distance
>>>>>>> main

def get_matrix(keys, default_val):
        m = {(s1, s2) : default_val() for s1 in keys() for s2 in keys()}
        return m
    
def lookup(m, split, op, base):
    ans = defaultdict(base)
    for k, v in m.items():
        ans[split(k)] = op(ans[split(k)], v)
    return ans

def plot_y_hist(folder, k, arr):
    y = arr[:, 0]
    pred = arr[:, 1]

    plt.hist(y, bins=50, alpha=0.5, label='y')
    plt.hist(pred, bins=50, alpha=0.5, label='pred')
    plt.legend()
    plt.title(str(k))
    plt.savefig(f"run_data/plots/{folder}/{k}_y_pred_histogram.png")
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filepath", type=str)
    args = parser.parse_args()
    checkpoint = torch.load(args.model_filepath)

    feature_extractor = MontezumaRamFeatureExtractor()
    test_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"), feature_extractor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
<<<<<<< HEAD
    model = DistanceNetwork(34, 1)
=======
    model = DistanceNetwork(34, 2)
>>>>>>> main
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device(device))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128)

    quantile = .05
    loss_fn = lambda error: quantile_loss(error, quantile, reduce='none')
    losses = []
    test_episodes = 30
<<<<<<< HEAD
    avg_size = 550
    test_size = int(avg_size * (avg_size + 1) * test_episodes / 256)
=======
    test_size = int(275 * 551 * test_episodes / 128)
>>>>>>> main
    keys = get_all_combos()
    y_m = get_matrix(keys, list)
    cnt_m = get_matrix(keys, int)
    total_loss_m = get_matrix(keys, int)
    
    with torch.no_grad():
        for X, predict_y, true_y, img in tqdm(itertools.islice(test_dataloader, 0, test_size), total=test_size):
            X = X.float().to(device)
            y = predict_y.to(device)
            pred = model(X).squeeze()
            loss = loss_fn(pred - y)
            for x, predict_y, true_y, pred, loss in zip(X, predict_y, true_y, pred, loss):
                k = parse_example(x)

                cnt_m[k] += 1
                y_m[k].append((predict_y.item(), pred.item()))
                total_loss_m[k] += loss.item()

    relevant_keys = [k for (k, v) in cnt_m.items() if v > 4000]
    y_m = {k : np.array(y_m[k]) for k in relevant_keys}

    avg_loss_m = {k : total_loss_m[k] / cnt_m[k] for k in relevant_keys}
    by_avg_loss = sorted(avg_loss_m.items(), key=lambda kv: kv[1], reverse=True)
    print('---------HIGH LOSS----------')
    for i in range(8):
        k, v = by_avg_loss[i]
        cnt = cnt_m[k]
        print(k)
        print('average loss: ', v, 'cnt: ', cnt)
        plot_y_hist("high_loss", k, y_m[k])

    print('---------LOW LOSS-----------')
    for i in range(1,9):
        k, v = by_avg_loss[-i]
        cnt = cnt_m[k]
        print(k)
        print('average loss: ', v, 'cnt: ', cnt)
        plot_y_hist("low_loss", k, y_m[k])

    # plot variance
<<<<<<< HEAD
    # variance_m = {}
    # for k, v in y_m.items():
    #     discounted_y = v[:, 0]
    #     pred = v[:, 1]
    #     undiscounted_y = v[:, 2]
    #     undiscounted_y_var = np.var(undiscounted_y)
    #     undiscounted_y_var_over_mean = undiscounted_y_var / np.mean(undiscounted_y)
    #     discounted_y_var = np.var(discounted_y)
    #     pred_var = np.mean([1/np.var(p) for p in pred])
    #     variance_m[k] = [undiscounted_y_var, undiscounted_y_var_over_mean, discounted_y_var, pred_var]
    # 

    # true_distance = [min(y_m[k][:, 2]) for k in relevant_keys]
    # avg_distance = [np.mean(y_m[k][:, 2]) for k in relevant_keys]
    # losses = [avg_loss_m[k] for k in relevant_keys]
    # total_loss_scaled = [total_loss_m[k] * 2e-4 for k in relevant_keys]
    # for idx, title in [(0, 'undiscounted_variance'), (1, 'undiscounted_variance_over_mean'), (2, 'discounted_variance'), (3, 'avg_prediction_uncertainty')]:
    #     arr = [variance_m[k][idx] for k in relevant_keys]
    #     plt.hist(arr, bins=50)
    #     plt.title(title)
    #     plt.savefig(f"run_data/plots/variance/{title}.png")
    #     plt.close('all')

    #     plt.scatter(arr, losses, alpha=0.4, s=total_loss_scaled)
    #     plt.title(f"{title} vs loss")
    #     plt.savefig(f"run_data/plots/variance/{title}_vs_loss.png")
    #     plt.close('all')

    #     plt.scatter(true_distance, arr, alpha=0.4)
    #     plt.title(f"true distance vs {title}")
    #     plt.xlabel("true distance")
    #     plt.ylabel(title)
    #     plt.savefig(f"run_data/plots/variance/true_distance_vs_{title}.png")
    #     plt.close('all')

    #     plt.scatter(avg_distance, arr, alpha=0.4)
    #     plt.title(f"avg distance vs {title}")
    #     plt.xlabel("avg distance")
    #     plt.ylabel(title)
    #     plt.savefig(f"run_data/plots/variance/avg_distance_vs_{title}.png")
    #     plt.close('all')

    # # plot cnt
    # plt.hist(cnt_m.values(), bins=50, log=True)
    # plt.title("counts")
    # plt.savefig("run_data/plots/cnt_histogram.png")
    # plt.close('all')
=======
    variance_m = {}
    for k, v in y_m.items():
        discounted_y = v[:, 0]
        pred = v[:, 1]
        undiscounted_y = v[:, 2]
        undiscounted_y_var = np.var(undiscounted_y)
        undiscounted_y_var_over_mean = undiscounted_y_var / np.mean(undiscounted_y)
        discounted_y_var = np.var(discounted_y)
        pred_var = np.mean([1/np.var(p) for p in pred])
        variance_m[k] = [undiscounted_y_var, undiscounted_y_var_over_mean, discounted_y_var, pred_var]
    
    avg_distance = [np.mean(y_m[k][:, 2]) for k in relevant_keys]
    true_distance = [min(y_m[k][:, 2]) for k in relevant_keys]
    losses = [avg_loss_m[k] for k in relevant_keys]
    total_loss_scaled = [total_loss_m[k] * 2e-4 for k in relevant_keys]
    for idx, title in [(0, 'undiscounted_variance'), (1, 'undiscounted_variance_over_mean'), (2, 'discounted_variance'), (3, 'avg_prediction_uncertainty')]:
        arr = [variance_m[k][idx] for k in relevant_keys]
        plt.hist(arr, bins=50)
        plt.title(title)
        plt.savefig(f"run_data/plots/variance/{title}.png")
        plt.close('all')

        plt.scatter(arr, losses, alpha=0.4, s=total_loss_scaled)
        plt.title(f"{title} vs loss")
        plt.savefig(f"run_data/plots/variance/{title}_vs_loss.png")
        plt.close('all')

        plt.scatter(true_distance, arr, alpha=0.4)
        plt.title(f"true distance vs {title}")
        plt.xlabel("true distance")
        plt.ylabel(title)
        plt.savefig(f"run_data/plots/variance/true_distance_vs_{title}.png")
        plt.close('all')

        plt.scatter(avg_distance, arr, alpha=0.4)
        plt.title(f"avg distance vs {title}")
        plt.xlabel("avg distance")
        plt.ylabel(title)
        plt.savefig(f"run_data/plots/variance/avg_distance_vs_{title}.png")
        plt.close('all')

    # plot cnt
    plt.hist(cnt_m.values(), bins=50, log=True)
    plt.title("counts")
    plt.savefig("run_data/plots/cnt_histogram.png")
    plt.close('all')
>>>>>>> main
