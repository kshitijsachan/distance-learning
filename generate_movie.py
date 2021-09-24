import gym, d4rl_atari, cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

def write_video(file_path, frames, fps):
    w, h = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()

D4RL_ENV_NAME = 'montezuma-revenge-expert-v4'
d4rl_env = gym.make(D4RL_ENV_NAME)
dataset = d4rl_env.get_dataset()
obs = np.squeeze(dataset['observations'])
cnt_length = len(str(len(obs) - 1))

for i, ob in tqdm(enumerate(obs)):
    im = Image.fromarray(ob)
    cnt = str(i).zfill(cnt_length)
    im.save(f"/screenshots/{cnt}.png")
# run: ffmpeg -i /screenshots/%6d.png -r 25 all_frames.mp4
# to speed it up: ffmpeg -i all_frames.mp4 -filter:v "setpts=PTS/3" all_frames3.mp4
