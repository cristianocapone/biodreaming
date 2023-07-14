"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Biologically Plausible Model-Based Reinforcement Learning in Recurrent Spiking Networks
    Authors: Anonymus
"""
import ffmpeg

import numpy as np
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

N = 256

vals = np.ones((N, 4))
vals[:, 0] = np.linspace(30/256, 1, N)
vals[:, 1] = np.linspace(30/256, 1, N)
vals[:, 2] = np.linspace(30/256, 1, N)
vscode_cm = ListedColormap(vals)

vals = np.ones((N, 4))
vals[:, 0] = np.linspace(30/256, .80, N)
vals[:, 1] = np.linspace(30/256, .05, N)
vals[:, 2] = np.linspace(30/256, .05, N)
vserror_cm = ListedColormap(vals)

vals = np.ones((N, 4))
vals[:, 0] = np.linspace(30/256, .05, N)
vals[:, 1] = np.linspace(30/256, .80, N)
vals[:, 2] = np.linspace(30/256, .05, N)
vscorrect_cm = ListedColormap(vals)

def default(var, val):
    return val if var is None else var

def dark_theme():
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor']='#1E1E1E'
    plt.rcParams['figure.facecolor']='#1E1E1E'

def kTrajectory (T, K = 3, Ar = (0.5, 2.0), Wr = (1, 2, 3, 5), offT = 0, norm = False):
    P = []

    for k in range (K):
        A = np.random.uniform (*Ar, size = 4)
        W = np.array (Wr) * 2. * np.pi
        F = np.random.uniform (0., 2. * np.pi, size = len (W))
        t = np.linspace (0., 1., num = T)

        p = 0.
        for a, w, f in zip (A, W, F):
            p += a * np.cos (t * w + f)

        P.append (p)
    P = np.array (P)

    # Here we normalize our trajectories
    P = P / np.max (np.abs (P), axis = 1).reshape (K, 1) if norm else P

    # Here we zero-out the initial offT entries of target
    P [:, :offT] = 0.

    return P

def kClock (T, K = 5):
    C = np.zeros ((K, T))

    for k, tick in enumerate (C):
        range = T // K

        tick [k * range : (k + 1) * range] = 1

    return C

# def build_ktarg(
#     dim : int = 100,
#     num : int = 3,
#     amp : Tuple[float, float] = (0.5, 2.0),
#     freq : List[float] = (1, 2, 3, 5),
# ) -> Tuple[Tensor, Tensor]:
    
#     a, b = amp

#     # * Build input
#     inps = torch.block_diag(*torch.ones(num, dim // num))
    
#     # * Build target output
#     amps = (b - a) * torch.rand(num, 1, 1) + a
#     freq : Tensor = torch.tensor(freq)
#     phis = torch.rand(num, 1, len(freq)) * 2 * torch.pi

#     time = torch.linspace(0, 2 * torch.pi, dim)
#     time = time.reshape(1, dim, 1).expand(num, dim, len(freq))

#     targ = amps * torch.cos(time * freq + phis)

#     return inps, targ.sum(-1)

def sfilter (seq, itau = 0.5):
    filt_seq = np.zeros (seq.shape)

    for t, s in enumerate (seq.T):
        filt_seq [:, t] = filt_seq [:, t - 1] * itau + s * (1. - itau) if t > 0 else seq [:, 0]

    return filt_seq

# @torch.no_grad()
# def expfilt(seq : Tensor, itau : float) -> Tensor:
#     out = torch.zeros_like(seq)
    
#     # Filter the output using readout time constant
#     for t, v in enumerate(seq):
#         out[t] = out[t - 1] * itau + (1 - itau) * v

#     return out

def dJ_rout (J_rout, targ, S_rout):
    Y = J_rout @ S_rout

    return (targ - Y) @ S_rout.T

def read (S, J_rout, itau_ro = 0.5):
    out = sfilter (S, itau = itau_ro)

    return J_rout @ out

def gaussian (x, mu = 0., sig = 1.):
    return np.exp (-np.power (x - mu, 2.) / (2 * np.power (sig, 2.)))

def shuffle(iter):
    rng_state = np.random.get_state()

    for a in iter:
        np.random.shuffle(a)
        np.random.set_state(rng_state)

    for a in range (len(iter)):
        np.random.shuffle ([0., 1.])

def make_movie(savename : str, folder : str = 'frames', fps : int = 10):
    (
        ffmpeg
        .input(f'{folder}/*.jpg', pattern_type = 'glob', framerate = fps)
        .output(f'{savename}.mp4')
        .run()
    )