import torch, pickle, argparse, ipdb, itertools, random
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import TripletLossDataset 
from utils import IterativeAverage, trajectories_generator
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor

def run_loop(model, dataset, episodes, device, batch_size, experiment_name):
    # loop size
    avg_length = 550
    positive_radius = 5
    num_batches = int((2 * positive_radius + 1) * avg_length * episodes / batch_size)

    # List[int], predicted distance
    positives = []
    negatives = []

    loss_fn = torch.nn.MSELoss()
    for anchor, pos, neg, img in tqdm(itertools.islice(dataset, 0, num_batches), total=num_batches):
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        d_pos = model(torch.cat((anchor, pos), 1)).squeeze()
        d_neg = model(torch.cat((anchor, neg), 1)).squeeze()

        positives.extend(d_pos.detach().cpu().numpy())
        negatives.extend(d_neg.detach().cpu().numpy())

    plt.hist(positives, bins=30, alpha=0.4, label='close')
    plt.hist(negatives, bins=30, alpha=0.4, label='far')
    plt.legend()
    plt.xlabel("predicted distance")
    plt.savefig(f"run_data/triplet_loss_accuracy_{experiment_name}.png")
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    # use different seed than training
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 128
    device = "cuda"
    train_data_path = "/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_train.pkl.gz"
    test_data_path = "/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"
    
    # set up dataloader, unpickle model/optimizer
    feature_extractor = MontezumaRamFeatureExtractor()
    train_data = DataLoader(TripletLossDataset(lambda: trajectories_generator(train_data_path), feature_extractor), batch_size=batch_size)
    test_data = DataLoader(TripletLossDataset(lambda: trajectories_generator(test_data_path), feature_extractor), batch_size=batch_size)

    model = DistanceNetwork(12, 1).to(device)
    model.load_state_dict(torch.load(args.model_path)['model'])

    run_loop(model, train_data, 100, device, batch_size, "train_data")
    run_loop(model, test_data, 30, device, batch_size, "test_data")
