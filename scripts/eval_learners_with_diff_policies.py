import torch, pickle, argparse, ipdb
import matplotlib.pyplot as plt

from tqdm import tqdm

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import DistanceDataset
from utils import IterativeAverage, trajectories_generator
from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor

def test_loop(model, dataset, batch_size=128, device="cuda", num_samples=1e7):
    # num_samples ~= 80 episodes
    num_samples = int(num_samples / batch_size)
    loss_fn = torch.nn.MSELoss()
    loop_loss = IterativeAverage()
    cnt = 0
    for X, y, _, _ in tqdm(torch.utils.data.DataLoader(dataset, batch_size), total=num_samples):
        cnt += 1
        X = X.float().to(device)
        y = y.float().to(device)
        pred = model(X).squeeze()
        ipdb.set_trace()
        loss = loss_fn(pred, y).item()
        loop_loss.add(loss)
        if cnt == num_samples:
            return loop_loss.avg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_path", type=str)
    parser.add_argument("--model_path", type=str)

    beginner_model_path = "/home/ksachan/data/beginner_model.pt"
    expert_model_path = "/home/ksachan/data/expert_model.pt"
    beginner_data_path = "/home/ksachan/data/monte_rnd_trajs/beginner_policy/monte_rnd_first_3000_trajs_test.pkl.gz"
    expert_data_path = "/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_test.pkl.gz"
    
    # set up dataloader, unpickle model/optimizer
    feature_extractor = MontezumaRamFeatureExtractor()
    beginner_data = DistanceDataset(lambda: trajectories_generator(beginner_data_path), feature_extractor)
    expert_data = DistanceDataset(lambda: trajectories_generator(expert_data_path), feature_extractor)

    device = "cuda"
    beginner_model = DistanceNetwork().to(device)
    expert_model = DistanceNetwork().to(device)
    random_model = DistanceNetwork().to(device)
    beginner_model.load_state_dict(torch.load(beginner_model_path)['model'])
    expert_model.load_state_dict(torch.load(expert_model_path)['model'])

    print("beginner model, beginner data", test_loop(beginner_model, beginner_data))
    print("beginner model, expert data", test_loop(beginner_model, expert_data))
    print("expert model, beginner data", test_loop(expert_model, beginner_data))
    print("expert model, expert data", test_loop(expert_model, expert_data))
    print("random model, beginner data", test_loop(random_model, beginner_data))
    print("random model, expert data", test_loop(random_model, expert_data))
