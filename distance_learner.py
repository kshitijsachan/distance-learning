import argparse, torch, ipdb, itertools, sys, os, random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import DistanceDataset
from utils import IterativeAverage, trajectories_generator


class DistanceLearner():
    def __init__(self, train_dataset, test_dataset, label_mapper, num_classes, savedir, learning_rate=1e-4, batch_size=128, epochs=1, device=None, train_episodes=100, test_episodes=30):
        self.epochs = epochs
        self.batch_size = batch_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = DistanceNetwork(input_dim=12, output_dim=num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_data = lambda: torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self.test_data = lambda: torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        # plotting/history variables
        self.savedir = savedir
        avg_length = 550
        positive_radius
        self.episodes_to_batches = lambda episodes : int((2 * positive_radius + 1) * avg_length * episodes / batch_size)
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.train_loss = [] 
        self.test_loss = [] 
        self.loss_fn = triplet_loss

    def triplet_loss(self, anchor, positive, negative):
        d_pos = torch.norm(anchor - positive, 2)
        d_neg = torch.norm(anchor - negative, 2)
        undershoot = torch.nn.functional.relu(d_pos - d_neg + self.margin)
        num_positive_triplets = torch.sum(undershoot > 1e-16)
        ipdb.set_trace()
        return torch.sum(undershoot, dim=0) / (num_positive_triplets + 1e-16)

    def train_loop(self):
        loop_loss = IterativeAverage()
        train_size = self.episodes_to_batches(self.train_episodes)
        for X, predict_y, true_y, img in tqdm(itertools.islice(self.train_data(), 0, train_size), total=train_size):
            # forward pass
            self.optimizer.zero_grad()
            X = X.float().to(self.device)
            y = predict_y.float().to(self.device)
            pred = self.model(X).squeeze()
            loss = self.loss_fn(pred, y)

            # backpropagate
            loss.backward()
            self.optimizer.step()

            # update plotting vars
            loop_loss.add(loss.item())
        self.train_loss.append(loop_loss.avg())

    def test_loop(self):
        loop_loss = IterativeAverage()
        test_size = self.episodes_to_batches(self.test_episodes)
        for X, predict_y, true_y, img in tqdm(itertools.islice(self.test_data(), 0, test_size), total=test_size):
            X = X.float().to(self.device)
            y = predict_y.float().to(self.device)
            pred = self.model(X).squeeze()
            loss = self.loss_fn(pred, y).item()
            loop_loss.add(loss)
        self.test_loss.append(loop_loss.avg())

    def run(self):
        self.test_loop()
        for i in range(self.epochs):
            print(f"Epoch {i+1}\n-------------------------------")
            self.train_loop()
            self.test_loop()
            self._plot_loss()
            torch.save({
                'model' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict()
                }, os.path.join(self.savedir, f"pytorch_model.pt"))

    def _plot_loss(self):
        num_epochs = list(range(1, len(self.train_loss) + 1))
        plt.plot(num_epochs, self.train_loss, label="train")
        num_epochs = list(range(len(self.test_loss)))
        plt.plot(num_epochs, self.test_loss, label="test")
        plt.legend()
        plt.xlabel("# epochs trained")
        plt.ylabel("average loss")
        plt.title(f"epoch_size={self.train_episodes}")
        plt.savefig(os.path.join(self.savedir, f"loss.png"))
        plt.close('all')

def discount_distance(distance):
    gamma = 0.97
    return (1 - gamma ** distance) / (1 - gamma)

def bucket_distance1(distance):
    # 3 classes
    # gamma = 0.97
    # distance = (1 - gamma ** distance) / (1 - gamma)
    if distance < 100:
        return 0
    elif distance < 400:
        return 1
    else:
        return 2


def bucket_distance2(distance):
    if distance < 50:
        return 0
    return 1

def bucket_distance3(distance):
    if distance < 100:
        return 0
    return 1

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
    if args.mdp_name == "monte-ram":
        from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
        feature_extractor = MontezumaRamFeatureExtractor()
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
        savedir = os.path.join(savedir, args.experiment_name + f"_seed={seed}")
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "run_command.txt"), "w") as f:
        f.write(' '.join(str(arg) for arg in sys.argv))

    train_data = TripletLossDataset(lambda: trajectories_generator(args.train_data_path), feature_extractor)
    test_data = TripletLossDataset(lambda: trajectories_generator(args.test_data_path), feature_extractor)
    learner = DistanceLearner(train_data, test_data, label_mapper, num_classes, savedir=savedir, device=args.device, epochs=args.num_epochs)
    learner.run()
