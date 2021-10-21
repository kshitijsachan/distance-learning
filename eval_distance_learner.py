import argparse, torch, ipdb
import numpy as np
from PIL import Image

from montezuma_ram_feature_extractor import MontezumaRamFeatureExtractor
from distance_network import DistanceNetwork
from dataset import DistanceDataset
from utils import trajectories_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filepath", type=str)
    args = parser.parse_args()
    checkpoint = torch.load(args.model_filepath)

    learning_rate = 1e-3
    feature_extractor = MontezumaRamFeatureExtractor()
    test_data = DistanceDataset(lambda: trajectories_generator("/home/ksachan/data/monte_rnd_good_trajs_test.pkl"), feature_extractor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistanceNetwork()
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128)
    loss_fn = torch.nn.MSELoss()

    losses = []
    with torch.no_grad():
        for i, (X, y, images) in enumerate(test_dataloader):
            print(i)
            if i > 100:
                break
            images = np.squeeze(np.concatenate(images, axis=2))
            X = X.float().to(device)
            y = y * y
            pred = model(X)
            loss = loss_fn(pred, y.float().to(device)).item()
            losses.append(loss)
            im = Image.fromarray(images[0]).convert("RGB")
            ipdb.set_trace()
            im.save(f"run_data/pred_{pred[0]}_actual_{y[0]}_loss_{loss}.png")




    
