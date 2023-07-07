
import gym
import pickle
import numpy as np

from os import path
from PIL import Image
from tqdm import trange
from argparse import ArgumentParser
from argparse import Namespace

from src.agent import AGEMONE
from src.config import Config

from collections import defaultdict

def main(args : Namespace):
    # * Initialization of environment
    env = gym.make(args.env, render_mode = args.render_mode, time_limit = args.timeout)

    agent   = AGEMONE(Config[args.env])
    planner = AGEMONE(Config[args.env])

    stats = defaultdict(list)

    for rep in trange(args.n_rep):
        agent.forget()
        planner.forget()

        reward_tot = []
        reward_fin = []

        for _ in trange(args.epochs):
            agent.reset()
            planner.reset()
            obs, info = env.reset(options = {'button_pressed' : True})

            # * §§§ Awake phase §§§
            r_tot = 0
            done, timeout = False, False

            while not done and not timeout:
                # * Agent action
                action, out = agent.step(obs['agent_target'], deterministic = True)

                # * Planner action

                # * Environment step
                obs, r_fin, done, timeout, info = env.step(action)

                r_tot += r_fin
            
            reward_tot.append(r_tot)
            reward_fin.append(r_fin)

        stats['reward_tot'].append(reward_tot)
        stats['reward_fin'].append(reward_fin)

            # * §§§ Dreaming phase §§§
            # for _ in range(args.num_dream):
            #     agent.reset()
            #     planner.reset()

            #     obs, info = env.reset(options = {'button_pressed' : True})

            #     for dream_t in range(args.dream_len):
            #         # * Agent action
            #         action, out = agent.step(obs['agent_target'], deterministic = True)

            #         # * Planner predicts the new observation
            #         _, _ = planner.step(obs, deterministic = True)
            #         Δ_obs_pred, r_pred = planner.prediction()

            #         obs += Δ_obs_pred

            #         agent.accumulate_error(r_pred)

            #     agent.update_J()
        
        # * Save agent
        agent.save(path.join(args.save_dir, f'agent_{args.env}_{str(rep).zfill(2)}.pkl'))

        # * Save planner
        planner.save(path.join(args.save_dir, f'planner_{args.env}_{str(rep).zfill(2)}.pkl'))

    print(stats)

    with open(path.join(args.save_dir, f'stats_{args.env}.pkl'), 'wb') as f:
        pickle.dump(stats, f)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-env', type = str, default = 'ButtonFood-v0', help = 'Environment to run')
    parser.add_argument('-n_rep', type = int, default = 10, help = 'Number of repetitions of the experiment')
    parser.add_argument('-epochs', type = int, default = 2000, help = 'Number of agent training iterations')
    parser.add_argument('-timeout', type = int, default = 1000, help = 'Max number of environment episodes to run')
    parser.add_argument('-num_dream', default = 0, help = 'Number of planner dreams')
    parser.add_argument('-dream_len', default = 50, help = 'Length of each planner dream')
    # parser.add_argument('-dream_lag', default = 50, help = 'Time step of dreams')
    parser.add_argument('-render_mode', type = str, default = None, choices = ['human', 'rgb_array', None], help = 'Rendering mode')

    parser.add_argument('-save_dir', type = str, default = 'data', help = 'Directory to save data')
    parser.add_argument('-load_dir', type = str, default = 'data', help = 'Directory to load data')


    args = parser.parse_args()

    main(args)

# env = ButtonFood(
#     render_mode='human'
# )

# imgs = []
# obs, info = env.reset(init_agent=(0.5, 0.5), init_button=(0.5, 0.9), init_target=(0.8, 0.3))

# for i in range(50):
#     if i < 12: action = (0.00, +0.03)
#     else:      action = (0.01, -0.02)

#     obs, r, done, info = env.step(action)

#     frame = env.render(render_mode = 'rgb_array')

#     imgs.append(Image.fromarray(frame))

#     # if done:
#     #     break

# env.close()

# for f, img in enumerate(imgs):
#     img.save(f'frames/render_example_{str(f).zfill(2)}.jpg')

# # imgs[0].save('render_example.gif', save_all=True, append_images=imgs[1:], fps=8, loop=0)
