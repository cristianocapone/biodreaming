"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Biologically Plausible Model-Based Reinforcement Learning in Recurrent Spiking Networks
    Authors: Anonymus
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def action2cat(action, n_actions = 3):
    # Here we reduce the action space from 6D to 3D thanks to the reduction
    # NOOP = FIRE | RIGHT = RIGHTFIRE | LEFT = LEFTFIRE
    action[action == 1] = 0 # Fire -> NoOp
    action[action == 4] = 2 # RightFire -> Right
    action[action == 5] = 3 # LeftFire -> Left

    # Convert action to be categorical: one-hot encoding.
    # ? NOTE: For indexing we would require numbers from
    # ?       from 0 to 2, however actions are coded as
    # ?       {0 : NOOP, 2 : RIGHT, 3 : LEFT}, we thus
    # ?       introduce an encoding & decoding map.
    act_idx_map = {0 : 0, 2 : 1, 3 : 2}

    action_cat = np.eye(n_actions)[[act_idx_map[int(act)] for act in action]]

    return action_cat

def act2cat(act):
    dicmap = {0 : 0, 1 : 0, 2 : 1, 3 : 2, 4 : 1, 5 : 2}

    return dicmap[act]

def cat2act(cat):
    dicmap = {0 : 0, 1 : 2, 2 : 3}

    return dicmap[cat]

def save_frames_as_gif(frames, path= "", filename='gym_animation_12.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def test_policy():

    env.reset()
    agent.reset()

    R = []
    RTOT=0

    agent.dH = np.zeros (agent.N)
    ram_all, r, done, _ = env.step (0)
    ram = ram_all[[49, 50, 51, 54]]

    for skip in range(20):

        ram_all, r, done, _ = env.step (0)

    time = 0
    frame = 0

    while not done:
        frame += 1
        time += 1
        ram = ram_all[[49, 50, 51, 54]]

        action, out = agent.step_det(ram/255)
        ram_all, r, done, _ = env.step ([cat2act(action)])

        RTOT +=r

    return RTOT

def import_ram(ram_all):
    ram = np.zeros((4,))
    ram[0] = int(ram_all[49])
    ram[1] = int(ram_all[50])
    ram[2] = int(ram_all[51])
    ram[3] = int(ram_all[54])
    return ram

def plot_rewards (REWARDS,REWARDS_MEAN,S,OUT,RAM,RAM_PRED,R,R_PRED,ENTROPY,filename = 'figure.png'):

    plt.figure(figsize=( 18, 18 ) )

    plt.subplot(421)
    plt.plot(REWARDS_MEAN)
    plt.ylabel('reward')
    plt.xlabel('iterations')

    plt.subplot(422)
    plt.plot(REWARDS)
    plt.ylabel('reward')
    plt.xlabel('iterations')

    plt.subplot(423)
    plt.plot(R)
    plt.plot(R_PRED)
    plt.ylabel('reward')
    plt.xlabel('time')

    plt.subplot(424)
    plt.imshow(1-np.array(S)[:,0:40].T,aspect='auto',cmap ='gray')

    plt.subplot(425)
    plt.plot(np.array(OUT))
    plt.ylim(-.1,1.1)
    plt.xlabel('time')
    plt.ylabel('policy')


    plt.subplot(426)
    plt.plot(np.array(RAM_PRED)[:,0])
    plt.plot(np.array(RAM)[:,0])
    plt.xlabel('time')
    plt.ylabel('RAM 0')

    plt.subplot(427)
    plt.plot(ENTROPY)#
    plt.ylabel('entropy')
    plt.xlabel('iterations')

    plt.subplot(428)
    plt.plot(np.array(RAM_PRED)[:,1])
    plt.plot(np.array(RAM)[:,1])
    plt.xlabel('time')
    plt.ylabel('RAM 1')

    plt.savefig(filename)
    plt.close()
    return

def plot_dram (DRAM,DRAM_PRED,R,R_PRED,MEAN_ERROR_RAM,MEAN_ERROR_R,filename = 'figure.png'):

    plt.figure(figsize=(6,10), dpi=72)

    plt.subplot(421)
    plt.plot(np.array(DRAM_PRED)[:,0])
    plt.plot(np.array(DRAM)[:,0])
    plt.xlim(0,100)

    plt.subplot(422)
    plt.plot(np.array(DRAM_PRED)[:,1])
    plt.plot(np.array(DRAM)[:,1])
    plt.xlim(0,100)

    plt.subplot(423)
    plt.plot(np.array(DRAM_PRED)[:,2])
    plt.plot(np.array(DRAM)[:,2])
    plt.xlim(0,100)

    plt.subplot(424)
    plt.plot(np.array(DRAM_PRED)[:,3])
    plt.plot(np.array(DRAM)[:,3])
    plt.xlim(0,100)

    plt.subplot(425)
    plt.plot(np.array(R_PRED))
    plt.plot(np.array(R))
    plt.xlim(0,100)

    plt.subplot(426)
    plt.plot(MEAN_ERROR_RAM)
    #plt.plot(ERROR_RAM)

    plt.subplot(427)
    plt.plot(MEAN_ERROR_R)
    #plt.plot(ERROR_R)

    plt.savefig(filename)
    plt.close()
    return

def plot_planning(REWS_PLAN,R,RAM_PLAN,RAM,S_agent,S_planner,t_skip,filename):

    plt.figure(figsize=(16,12), dpi=72)

    plt.subplot(331)
    plt.plot(np.array(REWS_PLAN))
    plt.plot(R)
    plt.plot([t_skip, t_skip],[-1, 1],'r--')
    plt.xlim(0,100)
    plt.xlabel('time')

    plt.subplot(332)
    plt.plot(np.array(RAM_PLAN)[:,0])
    plt.plot(np.array(RAM)[:,0])
    plt.plot([t_skip, t_skip],[0, 200],'r--')
    plt.ylabel('x_ball')
    plt.xlim(0,100)
    plt.xlabel('time')

    plt.subplot(337)
    plt.plot(np.array(RAM_PLAN)[:,1])
    plt.plot(np.array(RAM)[:,1])
    plt.plot([t_skip, t_skip],[0, 200],'r--')
    plt.xlabel('time')

    plt.xlim(0,100)

    plt.subplot(334)
    plt.plot(np.array(RAM_PLAN)[:,2])
    plt.plot([t_skip, t_skip],[0, 200],'r--')
    plt.plot(np.array(RAM)[:,2])
    plt.ylabel('y paddle')
    plt.xlim(0,100)
    plt.xlabel('time')

    plt.subplot(335)
    plt.plot(np.array(RAM_PLAN)[:,3])
    plt.plot(np.array(RAM)[:,3])
    plt.ylabel('y ball')
    plt.xlabel('time')
    plt.plot([t_skip, t_skip],[0, 200],'r--')

    plt.xlim(0,100)

    plt.subplot(333)
    plt.plot(np.array(RAM_PLAN)[:,0],np.array(REWS_PLAN),'o')
    plt.ylabel('rew')
    plt.xlabel('x_ball')

    plt.subplot(336)
    plt.imshow(1-np.array(S_agent)[:,0:40].T,aspect='auto',cmap ='gray')
    plt.ylabel('# neuron agent net')
    plt.xlabel('time')


    plt.subplot(339)
    plt.imshow(1-np.array(S_planner)[:,0:40].T,aspect='auto',cmap ='gray')
    plt.ylabel('# neuron model net')
    plt.xlabel('time')

    plt.savefig(filename )
    plt.close()
    return
