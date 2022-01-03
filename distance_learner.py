import argparse, torch, ipdb, itertools, sys, os, random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import TripletLossDataset
from dataset import DistanceDataset, D4rlDataset
from utils import IterativeAverage, trajectories_generator

class DistanceLearner():
    def __init__(self, train_dataset, test_dataset, savedir, learning_rate=1e-5, batch_size=32, epochs=1, device=None, train_episodes=100, test_episodes=30):
        self.epochs = epochs
        self.batch_size = batch_size
        self.get_train_data = get_train_data 
        self.get_test_data = get_test_data 
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = DistanceNetwork(input_dim=8, output_dim=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = self.triplet_loss
        
        # plotting/history variables
        self.savedir = savedir
        self.train_episodes = train_episodes
        self.train_loss = [] 
        self.test_loss = [] 
        self.loss_fn = self.triplet_loss
        self.margin = 10

    def triplet_loss(self, d_pos, d_neg):
        undershoot = torch.nn.functional.relu(d_pos - d_neg + self.margin)
        num_positive_triplets = torch.sum(undershoot > 1e-16)
        return torch.sum(undershoot, dim=0) / (num_positive_triplets + 1e-16)

    def train_loop(self, epoch, first_pass=False):
        loop_loss = IterativeAverage()

        train_data = DataLoader(self.get_train_data(self.train_episodes), batch_size=self.batch_size)
        for anchor, pos, neg, img in tqdm(train_data):
            # forward pass
            self.optimizer.zero_grad()
            anchor = anchor.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)
            d_pos = self.model(torch.cat((anchor, pos), 1)).squeeze()
            d_neg = self.model(torch.cat((anchor, neg), 1)).squeeze()
            loss = self.loss_fn(d_pos, d_neg)

            # backpropagate
            if not first_pass:
                loss.backward()
                self.optimizer.step()

            # update plotting vars
            loop_loss.add(loss.item())
        self.train_loss.append(loop_loss.avg())

    def test_loop(self):
        loop_loss = IterativeAverage()
        test_size = self.episodes_to_batches(self.test_episodes)
        test_data = DataLoader(self.get_test_data(self.test_episodes), batch_size=self.batch_size)
        for anchor, pos, neg, img in tqdm(test_data):
            anchor = anchor.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)
            d_pos = self.model(torch.cat((anchor, pos), 1)).squeeze()
            d_neg = self.model(torch.cat((anchor, neg), 1)).squeeze()
            loss = self.loss_fn(d_pos, d_neg)
            loop_loss.add(loss.item())
        self.test_loss.append(loop_loss.avg())

    def run(self):
        self.train_loop(0, True)
        if self.get_test_data is not None:
            self.test_loop()
        for i in range(1, self.epochs + 1):
            print(f"Epoch {i}\n-------------------------------")
            self.train_loop(i)
            if self.get_test_data is not None:
                self.test_loop()
            self._plot_loss()
            torch.save({
                'model' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict()
                }, os.path.join(self.savedir, f"pytorch_model.pt"))

    def _plot_loss(self):
        num_epochs = list(range(0, len(self.train_loss)))
        plt.plot(num_epochs, self.train_loss, label="train")
        if self.get_test_data is not None:
            num_epochs = list(range(len(self.test_loss)))
            plt.plot(num_epochs, self.test_loss, label="test")
        plt.legend()
        plt.xlabel("# epochs trained")
        plt.ylabel("average loss")
        plt.title(f"epoch_size={self.train_episodes}")
        plt.savefig(os.path.join(self.savedir, f"loss.png"))
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train distance metric on data")
    parser.add_argument("mdp_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--device", type=str) # usually one of cuda:0 or cuda:1
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)

    args = parser.parse_args()
    envs_dict = {
            'umaze' : 'maze2d-umaze-dense-v1',
            'umaze-sparse' : 'maze2d-umaze-v1',
            'medium-maze' : 'maze2d-medium-v1'
            }
    if args.mdp_name == "monte-ram":
        from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
        feature_extractor = MontezumaRamFeatureExtractor()
        train_data = lambda num_episodes: DistanceDataset(lambda: trajectories_generator(args.train_data_path), feature_extractor, num_episodes)
        test_data = lambda num_episodes: DistanceDataset(lambda: trajectories_generator(args.test_data_path), feature_extractor, num_episodes)
    elif args.mdp_name in envs_dict:
        train_data = lambda num_episodes: D4rlDataset(envs_dict[args.mdp_name], num_episodes)
        test_data = None
    else:
        parser.error(f"Invalid mdp name: {args.mdp_name}") 

    # set seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up save directory
    savedir = "run_data"
    if args.experiment_name is not None:
        savedir = os.path.join(savedir, args.experiment_name + f"_seed={seed}_quantile={args.quantile}")
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "run_command.txt"), "w") as f:
        f.write(' '.join(str(arg) for arg in sys.argv))

    learner = DistanceLearner(train_data, test_data, savedir=savedir, device=args.device, epochs=args.num_epochs)
    learner.run()
