import gzip, pickle, random
from tqdm import tqdm

def trajectories_generator(path):
    with open(path, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            print("finished reading data")
            pass

if __name__ == "__main__":
    DATASET_FILEPATH = '/home/ksachan/data/monte_rnd_good_trajs.pkl'
    gen = trajectories_generator(DATASET_FILEPATH)

    with open("/home/ksachan/data/monte_rnd_good_trajs_train.pkl", "wb") as f_train:
        with open("/home/ksachan/data/monte_rnd_good_trajs_test.pkl", "wb") as f_test:
            for i, traj in enumerate(gen):
                print(i)
                if random.random() <= 0.8:
                    pickle.dump(traj, f_train)
                else:
                    pickle.dump(traj, f_test)
