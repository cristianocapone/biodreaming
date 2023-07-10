
import gym
import pickle
import numpy as np

from os import path
from PIL import Image
from tqdm import trange
from argparse import ArgumentParser
from argparse import Namespace

from src.agent import Actor, Planner
from src.config import Config

from collections import defaultdict

def main(args : Namespace):
    # * Initialization of environment
    env = gym.make(
        args.env,
        render_mode = args.render_mode,
        time_limit = args.timeout,
        num_actions = args.num_actions,
    )

    agent   = Actor  (Config[args.env])
    planner = Planner(Config[args.env])

    stats = defaultdict(list)

    for rep in trange(args.n_rep):
        agent.forget()
        planner.forget()

        reward_tot = []
        reward_fin = []

        iterator = trange(args.epochs, desc = 'Episode reward: ---')
        for _ in iterator:
            agent.reset()
            planner.reset()
            obs, info = env.reset(options = {'button_pressed' : True})

            # * §§§ Awake phase §§§
            r_tot = 0
            done, timeout = False, False

            while not done and not timeout:
                # * Agent action
                action = agent.step(obs['agent_target'], deterministic = True)

                # * Planner action: prediction of next env state
                # Convert action to one-hot for concatenation with the observation
                planner_obs = np.concatenate((np.eye(args.num_actions)[action], obs['agent_target']))
                pred_state, pred_reward = planner.step(planner_obs, deterministic = True)

                # * Environment step
                obs, r_fin, done, timeout, info = env.step(action)

                # Update agents using the reward signal and planner using the prediction to
                # the next environment state and reward
                agent.accumulate_evidence(r_fin)
                planner.accumulate_evidence((pred_state, pred_reward), (obs['agent_target'], r_fin))
                planner.learn_from_evidence()

                r_tot += r_fin

            agent.learn_from_evidence()
            
            reward_tot.append(r_tot)
            reward_fin.append(r_fin)

            iterator.set_description(f'Episode reward {r_fin:.2f}')
 
            # * §§§ Dreaming phase §§§
            for _ in range(args.num_dream):
                agent.reset()
                planner.reset()

                obs, info = env.reset(options = {'button_pressed' : True})
                obs = obs['agent_target']

                for dream_t in range(args.dream_len):
                    # * Agent action
                    action = agent.step(obs, deterministic = True)

                    # * Planner predicts the new observation
                    planner_obs = np.concatenate((action, obs))
                    obs, reward = planner.step(planner_obs, deterministic = True)
                    
                    agent.accumulate_evidence(reward)

                agent.learn_from_evidence()

        stats['reward_tot'].append(reward_tot)
        stats['reward_fin'].append(reward_fin) 
        
        # * Save agent
        agent.save(path.join(args.save_dir, f'agent_{args.env}_{str(rep).zfill(2)}.pkl'))

        # * Save planner
        planner.save(path.join(args.save_dir, f'planner_{args.env}_{str(rep).zfill(2)}.pkl'))

    with open(path.join(args.save_dir, f'stats_{args.env}.pkl'), 'wb') as f:
        pickle.dump(stats, f)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-env', type = str, default = 'ButtonFood-v0', help = 'Environment to run')
    parser.add_argument('-n_rep', type = int, default = 10, help = 'Number of repetitions of the experiment')
    parser.add_argument('-epochs', type = int, default = 2000, help = 'Number of agent training iterations')
    parser.add_argument('-timeout', type = int, default = 1000, help = 'Max number of environment episodes to run')
    parser.add_argument('-num_dream', type = int, default = 0, help = 'Number of planner dreams')
    parser.add_argument('-dream_len', type = int, default = 50, help = 'Length of each planner dream')
    parser.add_argument('-num_actions', type = int, default = 8, help = 'Number of available discrete actions. Use 0 to trigger continuous control.')
    # parser.add_argument('-dream_lag', default = 50, help = 'Time step of dreams')
    parser.add_argument('-render_mode', type = str, default = None, choices = ['human', 'rgb_array', None], help = 'Rendering mode')

    parser.add_argument('-save_dir', type = str, default = 'data', help = 'Directory to save data')
    parser.add_argument('-load_dir', type = str, default = 'data', help = 'Directory to load data')


    args = parser.parse_args()

    main(args)