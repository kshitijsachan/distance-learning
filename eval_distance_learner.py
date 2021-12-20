import argparse, torch, ipdb, itertools, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor 
from distance_network import DistanceNetwork
from dataset import DistanceDataset
from utils import trajectories_generator, parse_example, get_all_combos
from distance_learner import bucket_distance3 as bucket_distance

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
    pred = [np.argmax(single_pred) for single_pred in arr[:, 1]]
    pred_var = [1 / np.var(single_pred) for single_pred in arr[:, 1]]
    undiscounted_y = arr[:, 2]

    plt.hist(y, bins=50)
    plt.title(str(k))
    plt.savefig(f"run_data/plots/{folder}/{k}_y.png")
    plt.close('all')
    plt.hist(pred)
    plt.title(str(k))
    plt.savefig(f"run_data/plots/{folder}/{k}_pred.png")
    plt.close('all')
    plt.hist(pred_var)
    plt.title(str(k))
    plt.savefig(f"run_data/plots/{folder}/{k}_pred_variance.png")
    plt.close('all')
    plt.hist(undiscounted_y, bins=50)
    plt.title(str(k))
    plt.savefig(f"run_data/plots/{folder}/{k}_undiscounted_y.png")
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filepath", type=str)
    args = parser.parse_args()
    checkpoint = torch.load(args.model_filepath)

    feature_extractor = MontezumaRamFeatureExtractor()
    test_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"), feature_extractor, bucket_distance)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistanceNetwork(34, 2)
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device(device))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    losses = []
    test_episodes = 30
    test_size = int(275 * 551 * test_episodes / 128)
    keys = get_all_combos()
    y_m = get_matrix(keys, list)
    cnt_m = get_matrix(keys, int)
    total_loss_m = get_matrix(keys, int)
    
    with torch.no_grad():
        for X, predict_y, true_y, img in tqdm(itertools.islice(test_dataloader, 0, test_size), total=test_size):
            X = X.float().to(device)
            y = predict_y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            for x, predict_y, true_y, pred, loss in zip(X, predict_y, true_y, pred, loss):
                k = parse_example(x)
                pred = pred.softmax(0)

                cnt_m[k] += 1
                y_m[k].append((predict_y.item(), pred.cpu().numpy(), true_y.item()))
                total_loss_m[k] += loss.item()

    relevant_keys = [k for (k, v) in cnt_m.items() if v > 4000]
    y_m = {k : np.array(y_m[k]) for k in relevant_keys}

    avg_loss_m = {k : total_loss_m[k] / cnt_m[k] for k in relevant_keys}
    by_avg_loss = sorted(avg_loss_m.items(), key=lambda kv: kv[1], reverse=True)
    print('---------HIGH LOSS----------')
    keys = [(('left-ladder', 0, True), ('middle-left', 0, True)),
            (('left-ladder', 0, True), ('bottom', 0, True)),
            (('top-middle', 1, False), ('top-middle', 1, False)),
            (('middle-middle', 0, False), ('middle-middle', 0, False)),
            (('middle-left', 0, True), ('bottom', 0, True)),
            (('middle-left', 0, True), ('middle-left', 0, True)),
            (('middle-right', 5, False), ('right-ladder', 5, False)),
            (('misc', 5, False), ('bottom', 5, False))]
    for i in range(8):
        k, v = by_avg_loss[i]
        cnt = cnt_m[k]
        print(k)
        print('average loss: ', v, 'cnt: ', cnt)
        plot_y_hist("high_loss", k, y_m[k])

    print('---------LOW LOSS-----------')
    keys = [(('rope', 0, False), ('left-ladder', 0, True)),
            (('middle-middle', 2, False), ('middle-left', 0, True)),
            (('middle-ladder', 0, False), ('left-ladder', 0, True)),
            (('right-ladder', 0, False), ('left-ladder', 0, True)),
            (('misc', 0, False), ('left-ladder', 0, True)),
            (('misc', 2, False), ('left-ladder', 0, True)),
            (('middle-ladder', 0, False), ('middle-left', 0, True)),
            (('middle-middle', 0, False), ('left-ladder', 0, True))]
    for i in range(1,9):
        k, v = by_avg_loss[-i]
        cnt = cnt_m[k]
        print(k)
        print('average loss: ', v, 'cnt: ', cnt)
        plot_y_hist("low_loss", k, y_m[k])

    # plot variance
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
