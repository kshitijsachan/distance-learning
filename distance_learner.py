import argparse, torch, ipdb, copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from distance_network import DistanceNetwork
from feature_extractor import FeatureExtractor
from dataset import DistanceDataset
from utils import trajectories_generator

def train_loop(epoch_size, dataloader, model, loss_fn, history, optimizer, device):
    epoch_loss = 0
    num_examples = 0
    with tqdm(total=epoch_size) as pbar:
        for X, y, _images in dataloader:
            pbar.update(len(X))

            # forward pass
            optimizer.zero_grad()
            X = X.float().to(device)
            y = y.float().to(device)
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)

            # Backpropogate
            loss.backward()
            optimizer.step()

            # return after epoch is complete 
            num_examples += len(X)
            epoch_loss += loss.item()
            if num_examples > epoch_size:
                average_loss = epoch_loss / num_examples
                print(f"train_loss: {average_loss:>3f}")
                history['train_loss'].append(average_loss)
                return

def test_loop(epoch_size, dataloader, model, loss_fn, history, device):
    epoch_loss = 0
    num_examples = 0
    with tqdm(total=epoch_size) as pbar:
        with torch.no_grad():
            for X, y, _images in dataloader:
                pbar.update(len(X))
                X = X.float().to(device)
                pred = model(X).squeeze()
                epoch_loss += loss_fn(pred, y.float().to(device)).item()
                num_examples += len(X)
                if num_examples > epoch_size:
                    average_loss = epoch_loss / num_examples
                    print(f"test loss: {average_loss:>3f}")
                    history['test_loss'].append(average_loss)
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train distance metric on data")
    parser.add_argument("mdp_name", type=str)

    args = parser.parse_args()
    if args.mdp_name == "monte-ram":
        from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
        feature_extractor = MontezumaRamFeatureExtractor()
        train_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_good_trajs_train.pkl"), feature_extractor)
        test_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_good_trajs_test.pkl"), feature_extractor)
    else:
        parser.error(f"Invalid mdp name: {args.mdp_name}") 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistanceNetwork().to(device)
    print(f"Using {device} device")
    print(model)

    # hyperparameters
    AVERAGE_EPISODE_SIZE = 500 * 501 / 2
    learning_rate = 1e-3
    batch_size = 128
    train_epoch_size = 5 * 10 ** 6 # num transitions, not num batches. ~40 episodes
    test_epoch_size = 1 * 10 ** 6 # ~8 episodes
    epochs = 8
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    history = {'train_loss' : [], 'test_loss' : []}

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_epoch_size, train_dataloader, model, loss_fn, history, optimizer, device)
        test_loop(test_epoch_size, test_dataloader, model, loss_fn, history, device)
    print('-' * 80)
    print("Saving model and plotting loss")
    print('-' * 80)
    torch.save({
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
        }, 'run_data/pytorch_model.pt')

    x_axis = [int(i * train_epoch_size / AVERAGE_EPISODE_SIZE) for i in range(1, epochs + 1)]
    plt.plot(x_axis, history['train_loss'], label='train_loss')
    plt.plot(x_axis, history['test_loss'], label='test_loss')
    plt.legend()
    plt.title('distance learning loss on absolute distance')
    plt.xlabel('num trajectories trained on')
    plt.ylabel('loss')
    plt.savefig('run_data/loss.png')

