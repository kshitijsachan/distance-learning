import argparse, torch, ipdb, copy, itertools, sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import DistanceDataset
from utils import IterativeAverage, trajectories_generator


class DistanceLearner():
    def __init__(self, train_dataset, test_dataset, savedir, learning_rate=1e-5, batch_size=128, epochs=1, device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.loss_fn = torch.nn.MSELoss()
        self.model = DistanceNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        # plotting/history variables
        self.savedir = savedir
        self.step_num = []
        self.train_loss = [] 
        self.test_loss = [] 

    def train_loop(self, num_batches):
        loop_loss = IterativeAverage()
        num_unique_train_examples = 0
        for X, y, _images, episode_num in tqdm(itertools.islice(self.train_dataloader, 0, num_batches), total=num_batches):
            for _ in range(self.epochs):
                # forward pass
                self.optimizer.zero_grad()
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                pred = self.model(X).squeeze()
                loss = self.loss_fn(pred, y)

                # backpropagate
                loss.backward()
                self.optimizer.step()

                # update plotting vars
                loop_loss.add(loss.item())
            num_unique_train_examples += len(X)
        self.step_num.append(num_unique_train_examples)
        self.train_loss.append(loop_loss.avg())

    def test_loop(self, num_batches):
        loop_loss = IterativeAverage()
        for X, y, _images, episode_num in tqdm(itertools.islice(self.test_dataloader, 0, num_batches), total=num_batches):
            X = X.float().to(self.device)
            y = y.float().to(self.device)
            pred = self.model(X).squeeze()
            loss = self.loss_fn(pred, y).item()
            loop_loss.add(loss)
        self.test_loss.append(loop_loss.avg())

    def run(self):
        train_loop_size = int(3e6 / self.batch_size) # ~24 episodes
        test_loop_size = int(1.5e6 / self.batch_size) # ~12 episodes

        num_loops = 25
        for i in range(num_loops):
            print(f"Loop {i+1}\n-------------------------------")
            self.train_loop(train_loop_size)
            self.test_loop(test_loop_size)
            self._plot_loss()
        torch.save({
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, os.path.join(self.savedir, f"pytorch_model_epoch={self.epochs}.pt"))


    def _plot_loss(self):
        step_num = np.cumsum(self.step_num)
        plt.plot(step_num, self.train_loss, label="train")
        plt.plot(step_num, self.test_loss, label="test")
        plt.legend()
        plt.xlabel("# steps trained")
        plt.ylabel("average loss")
        plt.savefig(os.path.join(self.savedir, f"step_loss_epoch={self.epochs}.png"))
        plt.close('all')

        average_episode_length = 500 * 501 / 2
        num_episodes = np.array(step_num) / average_episode_length
        plt.plot(num_episodes, self.train_loss, label="train")
        plt.plot(num_episodes, self.test_loss, label="test")
        plt.legend()
        plt.xlabel("# episodes trained")
        plt.ylabel("average loss")
        plt.savefig(os.path.join(self.savedir, f"episode_loss_epoch={self.epochs}.png"))
        plt.close("all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train distance metric on data")
    parser.add_argument("mdp_name", type=str)
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

    # set up save directory
    savedir = "run_data"
    if args.experiment_name is not None:
        savedir = os.path.join(savedir, args.experiment_name)
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "run_command.txt"), "w") as f:
        f.write(' '.join(str(arg) for arg in sys.argv))

    train_data = DistanceDataset(lambda: trajectories_generator(args.train_data_path), feature_extractor)
    test_data = DistanceDataset(lambda: trajectories_generator(args.test_data_path), feature_extractor)
    learner = DistanceLearner(train_data, test_data, savedir=savedir, device=args.device, epochs=args.num_epochs)
    learner.run()
