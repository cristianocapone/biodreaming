
import gym
import pickle
import numpy as np

from os import path
from tqdm import trange
from argparse import ArgumentParser
from argparse import Namespace

from scipy.stats import pearsonr

from src.agent import Actor, Planner
from src.config import Config
from src.monitor import Recorder

from collections import defaultdict

def main(args : Namespace):
    # * Initialization of environment
    env = gym.make(
        args.env,
        time_limit  = args.timeout,
        render_mode = args.render_mode,
        num_actions = args.num_actions,
    )

    agent   = Actor  (Config[f'{args.env}_Agent'])
    planner = Planner(Config[f'{args.env}_Planner'])
    monitor = Recorder(args.monitor, do_raise=args.strict_monitor)

    monitor.criterion = lambda episode : episode % args.monitor_freq == 0

    for rep in trange(args.n_rep):
        agent.forget()
        planner.forget()

        monitor.reset()

        reward_tot = []
        reward_corr = []

        env.register_step_callback(monitor)
        agent.register_step_callback(monitor)
        planner.register_step_callback(monitor)

        iterator = trange(args.epochs, desc = 'Episode reward: --- | Model-reward corr: ---')
        for episode in iterator:
            agent.reset()
            planner.reset()
            obs, info = env.reset(options = {
                    'button_pressed' : True,
                    'agent_init' : np.array((0.5, 0.5)),
                    # 'target_init' : np.array((0.8, 0.2)),
                    }
                )

            # * §§§ Awake phase §§§
            r_tot = 0
            done, timeout = False, False

            r_true_hist, r_pred_hist = [], [] 

            while not done and not timeout:
                state = obs['agent_target']

                # * Agent action
                # action = agent.step(state, deterministic = True, episode = episode)
                action = np.random.randint(8)

                # * Planner action: prediction of next env state
                pred_state, pred_reward = planner.step(
                                            state,
                                            # Convert action to one-hot for concatenation with the state
                                            action = np.eye(args.num_actions)[action],
                                            deterministic = True,
                                            episode = episode
                                        )

                # * Environment step
                obs, true_reward, done, timeout, info = env.step(action, episode = episode)

                # Update agents using the reward signal and planner using the prediction to
                # the next environment state and reward
                # agent.accumulate_evidence(r_fin)
                planner.accumulate_evidence((pred_state, pred_reward), (obs['agent_target'], true_reward))
                planner.learn_from_evidence()

                r_tot += true_reward

                r_true_hist.append(true_reward)
                r_pred_hist.append(pred_reward)

            # Commit monitor buffer after episode end to have clear episode separation in data
            monitor.commit_buffer()

            # agent.learn_from_evidence()
             
            r_corr = pearsonr(r_true_hist, r_pred_hist).statistic[0]
            msg = f'Episode reward {r_tot:.2f} | Model-reward corr. {r_corr:.2f}'
            iterator.set_description(msg)

            reward_tot.append(r_tot)
            reward_corr.append(r_corr)
 
            # * §§§ Dreaming phase §§§
            for _ in range(args.num_dream):
                agent.reset()
                planner.reset()

                obs, info = env.reset(options = {'button_pressed' : True})
                state = obs['agent_target']

                for dream_t in range(args.dream_len):
                    # * Agent action
                    action = agent.step(state, deterministic = True)

                    # * Planner predicts the new observation
                    state, reward = planner.step(
                                            state,
                                            action = np.eye(args.num_actions)[action],
                                            deterministic = True,
                                            # episode = episode
                                        )
                    
                    agent.accumulate_evidence(reward)

                agent.learn_from_evidence()

        monitor['reward_tot'].append(reward_tot)
        monitor['reward_corr'].append(reward_corr)
        
        # * Save agent
        agent.save(path.join(args.save_dir, f'agent_{args.env}_{str(rep).zfill(2)}.pkl'))

        # * Save planner
        planner.save(path.join(args.save_dir, f'planner_{args.env}_{str(rep).zfill(2)}.pkl'))

        # Save the monitored quantities
        filename = path.join(args.save_dir, f'stats_{args.env}.pkl')
        monitor.dump(filename)

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
    parser.add_argument('-monitor', type = str, nargs = '*', default = [], help = 'Path to monitor configuration for metric recording')
    parser.add_argument('-monitor_freq', type = int, default = 1, help = 'Episode Frequency for metric recording')
    
    parser.add_argument('-save_dir', type = str, default = 'data', help = 'Directory to save data')
    parser.add_argument('-load_dir', type = str, default = 'data', help = 'Directory to load data')

    parser.add_argument('--strict_monitor', action='store_true', default=False)

    args = parser.parse_args()

    main(args)