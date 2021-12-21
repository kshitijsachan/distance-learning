import pickle, gzip, itertools, ipdb

class IterativeAverage():
    def __init__(self):
        self.n = 0
        self.sum = 0

    def add(self, x):
        self.n += 1
        self.sum += x

    def avg(self):
        return self.sum / self.n

def trajectories_generator(path):
    with gzip.open(path, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            print("finished reading data")
            pass

def parse_example(example):
    example = example.tolist()
    size = int(len(example) / 2)
    s1, s2 = example[:size], example[size:]
    return _parse_state(s1), _parse_state(s2)

def _parse_state(state):
    xy = _parse_xy(state)
    lives = int(state[2])
    has_key = bool(state[3])
    # left_door_open = bool(state[7])
    # right_door_open = bool(state[8])
    # return (xy, lives, has_key, left_door_open, right_door_open)
    return (xy, lives, has_key)

def _parse_xy(state):
    if state[4]:
        return 'rope'

    normalized_x, normalized_y = state[0], state[1]
    x = normalized_x * 151 + 1
    y = normalized_y * 104 + 148
    if state[5]:
        if x < 36:
            return 'left-ladder'
        if x > 119:
            return 'right-ladder'
        return 'middle-ladder'

    if y >= 235:
        if x < 54:
            return 'top-left'
        if 63 < x < 89:
            return 'top-middle'
        if x > 98:
            return 'top-right'

    if y <= 165:
        return 'bottom'

    if 129 <= y <= 209:
        if x < 36:
            return 'middle-left'
        if 56 < x < 101:
            return 'middle-middle'
        if x > 119:
            return 'middle-right'
    return 'misc'

def get_all_combos():
        xy_pos = ['left-ladder', 'middle-ladder', 'right-ladder', 'rope', 'top-left', 'top-middle', 'top-right', 'bottom', 'middle-left', 'middle-middle', 'middle-right', 'misc']
        lives_left = list(range(6))
        has_key = [True, False]
        all_combos = lambda : itertools.product(xy_pos, lives_left, has_key)
        return all_combos
