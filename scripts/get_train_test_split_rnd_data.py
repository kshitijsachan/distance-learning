import gzip, pickle, random
from tqdm import tqdm

def trajectories_generator(path):
    with gzip.open(path, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            print("finished reading data")
            pass

if __name__ == "__main__":
    base_dir = "/home/ksachan/data/monte_rnd_trajs/expert_policy"
    DATASET_FILEPATH = base_dir + '/monte_rnd_last_3000_trajs.pkl.gz'
    gen = trajectories_generator(DATASET_FILEPATH)

    with gzip.open(base_dir + "/monte_rnd_last_3000_trajs_train.pkl.gz", "wb") as f_train:
        with gzip.open(base_dir + "/monte_rnd_last_3000_trajs_test.pkl.gz", "wb") as f_test:
            for traj in tqdm(gen, total=3000):
                if random.random() <= 0.8:
                    pickle.dump(traj, f_train)
                else:
                    pickle.dump(traj, f_test)
