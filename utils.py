import pickle

def trajectories_generator(path):
    with open(path, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            print("finished reading data")
            pass
