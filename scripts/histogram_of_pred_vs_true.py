import torch, pickle, math
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import DistanceDataset
from utils import IterativeAverage, trajectories_generator
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor

def average_loss(predictions):
    loss = IterativeAverage()
    for y in predictions:
        for pred in predictions[y]:
            loss.add((y - pred) ** 2)
    print("average loss", loss.avg())

def plot_true_histogram(predictions):
    plt.title("distance between training examples")
    plt.xlabel("distance")
    plt.ylabel("# examples")
    plt.plot(list(predictions.keys()), [len(predictions[k]) for k in predictions])
    plt.savefig("/home/ksachan/git-repos/distance-learning/run_data/true_histogram.png")
    plt.close("all")


def plot_predicted_histogram(predictions, low, high):
    pred = []
    for y in range(low, high + 1):
        pred += predictions[y]
    plt.hist(pred, bins=100)
    plt.title(f"histogram of predictions for y in [{low}, {high}]")
    plt.xlabel("predicted distance")
    plt.ylabel("frequency")
    plt.savefig(f"/home/ksachan/git-repos/distance-learning/run_data/pred_histogram_low={low}_high={high}.png")
    plt.close("all")


def plot_error_as_function_of_distance(predictions):
    errors = []
    std_devs= []
    for y in predictions:
        losses = [(y - pred) ** 2 for pred in predictions[y]]
        errors.append(np.mean(losses))
        std_devs.append(math.sqrt(np.var(losses)))
    plt.errorbar(list(predictions.keys()), errors, yerr=std_devs)
    plt.savefig("/home/ksachan/git-repos/distance-learning/run_data/pred_histogram_errors.png")
    plt.close("all")

if __name__ == "__main__":
    train_data_path = "/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"
    model_path = "/home/ksachan/data/expert_model.pt"
    pickle_dump_file = "/home/ksachan/git-repos/distance-learning/run_data/predicted_histograms.pkl"
    device = "cuda"
    batch_size = 128
    
    # set up dataloader, unpickle model/optimizer
    feature_extractor = MontezumaRamFeatureExtractor()
    test_data = DistanceDataset(lambda: trajectories_generator(train_data_path), feature_extractor)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    model = DistanceNetwork().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    predictions = defaultdict(list)

    i = 0
    for X, y, _images, _ep_num in tqdm(test_dataloader, total=int(1e7/batch_size)):
        i += 1
        X = X.float().to(device)
        y_s = y.float().to(device)
        preds = model(X).squeeze()
        for y, pred in zip(y_s, preds):
            predictions[y.item()].append(pred.item())
        if i > int(1e7 / batch_size):
            break

    ipdb.set_trace()

    with open(pickle_dump_file, "wb") as f:
        pickle.dump(predictions, f)
