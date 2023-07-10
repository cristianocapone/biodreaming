
import gym
import pickle
import numpy as np
from os import path

from os import path
from PIL import Image
from tqdm import trange
from argparse import ArgumentParser
from argparse import Namespace

from src.agent import AGEMONE
from src.config import Config

from collections import defaultdict

def main(args):
    # * Initialization of environment
    env = gym.make(
        args.env,
        render_mode = args.render_mode,
        time_limit = args.timeout,
        num_actions = args.num_actions,
    )

    frames = []
    agent = AGEMONE.load(path.join(args.load_dir, f'{args.model}_{args.env}_00.pkl'))

    for _ in range(args.num_episodes):
        agent.reset()
        obs, info = env.reset(options = {'button_pressed' : True})

        # * §§§ Awake phase §§§
        done, timeout = False, False

        while not done and not timeout:
            # * Agent action
            action, out = agent.step(obs['agent_target'], deterministic = True)

            # * Environment step
            obs, r_fin, done, timeout, info = env.step(action)

            frame = env.render(render_mode = args.render_mode)
            
            if args.render_mode == 'rgb_array':
                frames.append(Image.fromarray(frame))

    env.close()

    # Produce the final output video
    if args.render_mode == 'rgb_array':
        frames[0].save(path.join(args.save_dir, 'recorded.gif'), save_all=True, append_images=frames[1:], duration=1/30, loop=0)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-env', type = str, default = 'ButtonFood-v0', help = 'Environment to run')
    parser.add_argument('-num_episodes', type = int, default = 1, help = 'Number of episodes to record')
    parser.add_argument('-timeout', type = int, default = 100, help = 'Max number of environment episodes to run')
    parser.add_argument('-num_actions', type = int, default = 8, help = 'Number of available discrete actions. Use 0 to trigger continuous control.')
    parser.add_argument('-render_mode', type = str, default = 'rgb_array', choices = ['human', 'rgb_array', None], help = 'Rendering mode')

    parser.add_argument('-model', type = str, default='agent', help = 'Model name')
    parser.add_argument('-seed', type = int, default = 0, help = 'Random seed')
    parser.add_argument('-save_dir', type = str, default = 'data', help = 'Directory to save data')
    parser.add_argument('-load_dir', type = str, default = 'data', help = 'Directory to load data')

    args = parser.parse_args()

    main(args)