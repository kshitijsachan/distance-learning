import gzip, pickle, ipdb, itertools, os
from pathlib import Path
from PIL import Image
import numpy as np

def trajectories_generator(path):
    with gzip.open(path, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            print("finished reading data")
            pass

def load_trajectories(path, starts, num_trajs):
    '''
    Returns a list of list of trajectories from gzip pickle file.
    Args:
    path (str): filepath of pkl file containing trajectories
    starts (list[int]): start position of a single list of trajectories
    num_trajs (int): number of trajectories to return after each start period
    Returns:
    (list of trajectories)
    '''
    assert(len(starts) > 0)
    print(f"[+] Loading trajectories from file '{path}'")
    relative_starts = [starts[0]]
    for i in range(1, len(starts)):
        diff = starts[i] - starts[i - 1]
        assert(diff > num_trajs)
        relative_starts.append(diff)
    print(relative_starts)
    gen = trajectories_generator(path)
    trajs = []
    for start in relative_starts:
        traj = []
        for _ in range(start):
            next(gen)
        for _ in range(num_trajs):
            traj.append(next(gen))
        trajs.append(traj)
    # trajs = [list(itertools.islice(gen, start, num_trajs)) for start in relative_starts]
    return trajs

DATASET_FILEPATH = '/home/ksachan/data/monte_rnd_trajs/expert_policy/monte_rnd_last_3000_trajs_train.pkl.gz'
SCREENSHOTS_DIR = '/home/ksachan/data/rnd_screenshots'
MOVIES_DIR = '/home/ksachan/data/rnd_movies'
Path(MOVIES_DIR).mkdir(parents=True, exist_ok=True)
NUM_ENVS = 32
single_env_starts = [0]
starts = [start * NUM_ENVS for start in single_env_starts] 
num_trajs = 100

for start,episodes in zip(single_env_starts, load_trajectories(DATASET_FILEPATH, starts, num_trajs)):
    ffmpeg_screenshots_dir = f"{SCREENSHOTS_DIR}/start_episode_{start}"
    ffmpeg_movie_filepath = f"{MOVIES_DIR}/start_episode_{start}.mp4"
    Path(ffmpeg_screenshots_dir).mkdir(parents=True, exist_ok=True)
    frame_count = 0
    for episode in episodes:
        for ram, ob in episode:
            ob = np.squeeze(ob).astype(np.uint8)
            im = Image.fromarray(ob)
            im.save(f"{ffmpeg_screenshots_dir}/{str(frame_count).zfill(6)}.png")
            frame_count += 1

    os.system(f"ffmpeg -y -i {ffmpeg_screenshots_dir}/%6d.png -r 25 {ffmpeg_movie_filepath}")
    # os.system(f"ffmpeg -y -i {ffmpeg_movie_filepath} -filter:v \"setpts=PTS/2\" {ffmpeg_movie_filepath}")
# run: ffmpeg -i /screenshots/%6d.png -r 25 all_frames.mp4
# to speed it up: ffmpeg -i all_frames.mp4 -filter:v "setpts=PTS/3" all_frames3.mp4
