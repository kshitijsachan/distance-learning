import gzip, ipdb, pickle
from tqdm import tqdm

def load_trajs(path, max_limit, skip=0):
    with gzip.open(path, 'rb') as f:
        for _ in range(skip):
            traj = pickle.load(f)

        try:
            for _ in tqdm(range(max_limit)):
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass


if __name__ == "__main__":
    with open("monte_rnd_first_3000_trajs.pkl", "wb") as f:
        for traj in load_trajs("/home/ksachan/data/monte_rnd_trajs/monte_rnd_full_trajectories.pkl.gz", 3000):
            pickle.dump(traj, f)
