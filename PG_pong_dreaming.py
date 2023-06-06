"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Towards biologically plausible Dreaming and Planning in recurrent spiking networks
    Authors: Anonymus
"""

import gym
from copy import deepcopy
from random import randint
import pickle
import numpy as np
from matplotlib import animation
from functions import action2cat, act2cat, cat2act, save_frames_as_gif, plot_rewards, plot_dram
from functions import import_ram, plot_planning
import matplotlib.pyplot as plt
import os.path
import os
from tqdm import trange
from optimizer import Adam
from argparse import ArgumentParser

#if_dream = 0

parser = ArgumentParser()
parser.add_argument('-if_dream', required = False,  type = int, dest = 'if_dream', help = 'to dream or not to dream',default = 0)
par_inp = vars(parser.parse_args())

if_dream = par_inp['if_dream']

folder = "results_0"

start_learn = 1*50

# Check whether the specified path exists or not
isExist = os.path.exists(folder)
if not isExist:
  # Create a new directory because it does not exist
  os.makedirs(folder)
  print("The new directory is created!")

act_factor = 5.

for repetitions in range(10):

    N_ITER = 50*40
    TIMETOT = 100

    env = gym.make('Pong-ramDeterministic-v0', difficulty = 0)

    print (f'Pong: Observation space: {env.observation_space}')
    print (f'Pong: Action Meaning: {env.unwrapped.get_action_meanings()}')

    from agent import BasicPongAgent
    from agent import AGEMO
    from config import PONG_V4_PAR_I4 as par

    plt.rcParams.update({'font.size': 14})

    train_par = {'epochs'    : par['epochs'],
                 'epochs_out' : par['epochs_out'],
                 'clump'   : par['clump'], 'feedback'  : par['feedback'],
                 'verbose' : par['verbose'], 'rank' : par['rank']}
    par["I"]= 4

    agent = AGEMO(par)

    par["I"]= 3+4
    par["O"]= 3

    par["tau_ro"] = 2.*par["dt"]#OCCHIO era 2.*
    planner = AGEMO(par)

    alpha_rout = agent.par['alpha_rout']

    plt.figure()

    # Erase both the Jrec and the Jout
    agent.forget()
    # Reset agent internal variables
    agent.reset()

    count = -1
    agent.Jout = np.random.normal(0,.1,size=(agent.O,agent.N))
    agent.J = np.random.normal(0,1./np.sqrt(agent.N),size=(agent.N,agent.N))#*=0

    agent.adam_rec = Adam (alpha = 0.001, drop = .99, drop_time = 10000)
    agent.adam_out = Adam (alpha = 0.001, drop = .99, drop_time = 10000)

    eta_factor_r = 0.2
    planner.adam_out_s = Adam (alpha = 0.002, drop = .99, drop_time = 10000) #0.01
    planner.adam_out_r = Adam (alpha = 0.002*eta_factor_r, drop = .99, drop_time = 10000) #0.005
    planner.adam_rec = Adam (alpha = 0.004, drop = .99, drop_time = 10000)

    planner.Jout =np.random.normal(0,.1,size=(agent.O,agent.N))
    planner.J = np.random.normal(0,1./np.sqrt(agent.N),size=(agent.N,agent.N))#*=0
    planner.Jout_s_pred = np.zeros((agent.I,agent.N))
    planner.Jout_r_pred = np.zeros((1,agent.N))

    planner.dJ_aggregate = 0
    planner.dJout_s_aggregate = 0
    planner.dJout_r_aggregate = 0

    REWARDS = []
    REWARDS_MEAN = []
    REWARDS_STANDARD_MEAN = []
    ENTROPY = []

    ERROR_RAM = []
    ERROR_R = []

    MEAN_ERROR_RAM = []
    MEAN_ERROR_R = []


    S = []

    agent.dJ_aggregate=0
    agent.dJout_aggregate=0
    planner.state = 0

    for iteration in trange(N_ITER):

        env.reset()
        agent.reset()
        planner.reset()

        S_planner = []
        S_agent = []

        R = []
        R_PRED = []

        RAM = []
        RAM_PRED = []
        DRAM_PRED = []
        DRAM = []

        PLANNER_STATES = []

        agent.dH = np.zeros (agent.N)
        planner.dH = np.zeros (agent.N)

        RTOT = 0
        vx_old = 0

        agent.dJfilt =0
        agent.dJfilt_out = 0
        ram_all, r, done, _ = env.step (0)
        ram = np.zeros((4,))
        ram = import_ram(ram_all)
        ram_old = ram


        ######### AWAKE PHASE ##########

        for skip in range(20):
            act_vec = np.zeros((3,))
            act_vec = act_vec*0
            act_vec[0]=1

            _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )
            ds_pred,r_pred = planner.prediction()

            planner.state = ds_pred + ram

            S_planner.append(planner.S[:])
            S_agent.append(agent.S[:])

            ram_all, r, done, _ = env.step (0)

            ram_old = ram
            ram = np.zeros((4,))
            ram = import_ram(ram_all)

            PLANNER_STATES.append( planner.state_out )
            RAM_PRED.append( planner.state )
            RAM.append( ram )
            R += [r]
            R_PRED += [r_pred]

            dram = ram - ram_old
            dram[np.abs(dram)>30]=0.

            planner.learn_model(ds_pred,r_pred,dram,r)
            planner.model_update()

            DRAM.append(dram)
            DRAM_PRED.append(ds_pred)

        frame = 0
        ifplot = 1
        entropy=0

        OUT = []

        r_learn = 0

        while not done and frame<TIMETOT:

            frame += 1
            ram_old = ram

            action, out = agent.step_det(ram/255)
            act_vec = np.copy(out)

            act_vec = act_vec*0
            act_vec[action]=1

            _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )
            ds_pred,r_pred = planner.prediction()

            PLANNER_STATES.append( planner.state_out )

            S_planner.append(planner.S[:])
            S_agent.append(agent.S[:])

            ram_all, r, done, _ = env.step ([cat2act(action)])

            if_learn=0
            if iteration > start_learn:
                if_learn=1

            agent.learn_error(r*if_learn)

            ram = np.zeros((4,))
            ram = import_ram(ram_all)

            entropy+=agent.entropy

            dram = ram-ram_old
            dram[np.abs(dram)>30]=0.

            r_learn = r_learn*.5 + r

            planner.learn_model(ds_pred,r_pred,dram,r_learn)
            planner.model_update()

            planner.state = ram_old+ds_pred

            RAM_PRED.append(planner.state)
            RAM.append(ram)

            OUT.append(out)

            RTOT +=r
            R += [r]
            R_PRED += [r_pred]
            DRAM_PRED.append(ds_pred)
            DRAM.append(dram)

        REWARDS.append(RTOT)
        ENTROPY.append(entropy)
        ERROR_RAM.append(np.std(np.array(DRAM)-np.array(DRAM_PRED),axis=0))
        ERROR_R.append( np.std( np.array(R)-np.array(R_PRED) ) )


        if (iteration%1==0)&(iteration>0):
            agent.update_J(r)

        if (iteration%50==0)&(iteration>0):

            REWARDS_MEAN.append(np.mean(REWARDS[-50:]))
            plot_rewards(REWARDS,REWARDS_MEAN,S_agent,OUT,RAM,RAM_PRED,R,R_PRED,ENTROPY,filename = os.path.join(folder, 'rewards_dynamics_r0_initrand_aggr_ifdream_' + str(if_dream) + '.png') )
            np.save(os.path.join(folder,"rewards_" + str(repetitions) + "if_dream_" + str(if_dream) + ".npy"), REWARDS_MEAN)

            MEAN_ERROR_RAM.append(np.mean(np.array(ERROR_RAM)[-50:,:],axis=0))
            MEAN_ERROR_R.append(np.mean(np.array(ERROR_R)[-50:]))

            plot_dram(DRAM,DRAM_PRED,R,R_PRED,MEAN_ERROR_RAM,MEAN_ERROR_R,filename= os.path.join(folder, 'planning_dram_fit.png'))

        ######### DREAMING PHASE ##########

        plot_dream_every = 50

        for dream_times in range(if_dream):

            RAM_PLAN = []
            REWS_PLAN = []
            S_agent = []
            S_planner = []

            env.reset()
            agent.reset()
            planner.reset()

            ram_all, r, done, _ = env.step (0)
            ram = import_ram(ram_all)
            t_skip = 20

            for skip in range(t_skip):

                ram_all, r, done, _ = env.step (0)
                RAM_PLAN.append(ram_all[[49, 50, 51, 54]])

                act_vec = np.copy(out)
                act_vec = act_vec*0
                act_vec[0]=1

                _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )
                ds_pred,r_pred = planner.prediction()
                ram = import_ram(ram_all)
                REWS_PLAN.append(r_pred)

                S_planner.append(planner.S[:])
                S_agent.append(agent.S[:])

            time_dream = 50

            for plannng_steps in range(time_dream):
                agent.dH = np.zeros (agent.N)
                planner.dH = np.zeros (agent.N)

                action, out = agent.step_det(ram/255)

                act_vec = np.copy(out)
                act_vec = act_vec*0
                act_vec[action]=1

                _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )
                ds_pred,r_pred = planner.prediction()

                S_planner.append(planner.S[:])
                S_agent.append(agent.S[:])

                ram = ram + ds_pred

                if_learn=0
                if iteration > start_learn:
                    if_learn=1

                agent.learn_error(r_pred*if_learn)

                RAM_PLAN.append(ram)
                REWS_PLAN.append(r_pred)
            agent.update_J(r_pred)

            if (iteration%50==0)&(dream_times==0):
                plot_planning(REWS_PLAN,R,RAM_PLAN,RAM,S_agent,S_planner,t_skip,filename = os.path.join(folder, 'planning.png'))

    agent.save (os.path.join(folder,'model_PG_out_r0_initrand_aggr' + str(if_dream) + '_60''.py') )
