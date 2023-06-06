"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Biologically Plausible Model-Based Reinforcement Learning in Recurrent Spiking Networks
    Authors: Anonymus
"""

import gym
from copy import deepcopy
from random import randint

import pickle
import numpy as np

from matplotlib import animation

import matplotlib.pyplot as plt
import os.path
import os

from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter


folder = "results_0"

dream_vals = [0,1]
n_rep = [10,10]
step_vals = [0,0]
n_in = [0,0]

labels = [ 'not dreaming' , 'dreaming']

cmap = plt.get_cmap('copper')
colors = [cmap(i) for i in np.linspace(0, 1, len(step_vals))]

fig,ax = plt.subplots(1,1,figsize=(3.5,3.5))

final_rew = []
final_rew_sem = []

count=-1
for n_pred_steps in step_vals:

    count+=1
    if_dream = dream_vals[count]

    print(n_pred_steps)

    REWARDS = []
    for repetitions in range(n_in[count] , n_rep[count] ):
        reward = np.load(os.path.join(folder,"rewards_" + str(repetitions) + "if_dream_" + str(if_dream) + ".npy"))#_not_clumped_rank" + str(rank) + "
        plt.plot( savgol_filter(reward, 9, 3)   ,color = colors[count],lw=.25)
        REWARDS.append( savgol_filter(reward, 9, 3) )

    median = np.median(np.array(REWARDS),axis=0)
    sem = np.std(np.array(REWARDS),axis=0)/np.sqrt(n_rep[count])

    final_rew.append(median[-1])
    final_rew_sem.append(sem[-1])

    print(final_rew)

    ax.fill_between(range(len(median)), median+sem, median-sem,color=colors[count],alpha=0.1)
    ax.plot(np.percentile(np.array(REWARDS),50,axis=0),linestyle = 'dashed',linewidth=2.0,color=colors[count],label=labels[count])


    prct_80 = np.percentile(np.array(REWARDS),80,axis=0)
    yhat = savgol_filter(prct_80, 9, 3) # window size 51, polynomial order 3


    ax.plot(np.mean(np.array(REWARDS)),linestyle = 'dashed',linewidth=2.0,color=colors[count],label=labels[count])
    ax.plot( yhat ,linestyle = 'solid',linewidth=2.0,color=colors[count])

plt.plot( [0,39],[0,0] ,'r--')

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["bottom"].set_bounds(0, 40)
ax.spines["left"].set_bounds(-2., 1)

ax.set_xlabel('frames (x5000)')
ax.set_ylabel('reward (pong)')
#ax.legend()
plt.tight_layout()
fig.savefig(folder + '/comparison_10.png', dpi=600 )
fig.savefig(folder + '/comparison_10.eps')
