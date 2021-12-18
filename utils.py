import pickle, gzip

class IterativeAverage():
    def __init__(self):
        self.n = 0
        self.sum = 0

    def add(self, x):
        self.n += 1
        self.sum += x

    def avg(self):
        return self.sum / self.n

def trajectories_generator(path, num_skip):
    with gzip.open(path, 'rb') as f:
        try:
            for _ in range(num_skip):
                traj = pickle.load(f)
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            print("finished reading data")
            pass
