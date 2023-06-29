"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Towards biologically plausible Dreaming and Planning in recurrent spiking networks
"""

import numpy as np
import src.utils as ut
from optimizer import Adam, SimpleGradient
from tqdm import trange
import random

import matplotlib.pyplot as plt

class BasicPongAgent:
    ball_Yrange = (46, 205)
    play_Yrange = (38, 203)

    ball_Xend = 200

    action_map = {'NOOP' : 1, 'UP' : 2, 'DOWN' : 3}
    invact_map = {1 : 'NOOP', 2 : 'UP', 3 : 'DOWN'}

    def __init__(self):
        self.last_action = self.action_map['NOOP']
        self.curr_action = self.action_map['NOOP']

        self.frames = 0

        self.bmin, self.bmax = self.ball_Yrange
        self.pmin, self.pmax = self.play_Yrange

        self.last_by, self.last_bx = 0, 0

    def __call__(self, ram):
        self.frames += 1

        _, _, _, py, bx, by = self.extract_info(ram)

        #print(bx)

        vx = int(bx) - self.last_bx
        vy = int(by) - self.last_by

        self.last_bx, self.last_by = bx, by

        # Correct out-of-field value of by
        if by < self.bmin: by = .5 * (self.bmax + self.bmin)

        # Recast player and ball Y pos in range [0, 1]
        by = (by - self.bmin) / (self.bmax - self.bmin)
        py = (py - self.pmin) / (self.pmax - self.pmin)

        # If vx is negative it mean the ball is headed towards the enemy,
        # so we reposition at the center

        tx = (190 - bx)/(vx)
        y_pred = by + vy*tx

        if vx < 0 | np.isnan(y_pred):
            return self.action_map['NOOP']
        final_y = ((self.ball_Xend - bx) * vy / (vx + 1e-6)) % self.bmax
        self.state = (bx, self.last_bx, vx, by, self.last_by, vy, final_y)

        if bx>130:
            if   py > by + vy*0.00 + .05: return self.action_map['UP']
            elif py < by + vy*0.00 - .05: return self.action_map['DOWN']
            else:               return self.action_map['NOOP']
        else:
            return self.action_map['NOOP']

    def predict(self, ram):
        return self(ram)

    def extract_info(self, ram):
        cpu_score = ram[13]      # computer/ai opponent score
        player_score = ram[14]   # your score
        cpu_paddle_y = ram[50]     # Y coordinate of computer paddle
        player_paddle_y = ram[51]  # Y coordinate of your paddle
        ball_x = ram[49]           # X coordinate of ball
        ball_y = ram[54]           # Y coordinate of ball

        return (cpu_score, player_score, cpu_paddle_y, player_paddle_y, ball_x, ball_y)

class AGEMO:

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['N'], par['I'], par['O'], par['T']

        self.dt = par['dt']

        self.itau_m = np.exp (-self.dt / par['tau_m'])
        self.itau_s = np.exp (-self.dt / par['tau_s'])
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])
        self.itau_star = np.exp (-self.dt / par['tau_star'])

        self.dv = par['dv']

        self.hidden = par['hidden_steps'] if 'hidden_steps' in par else 1

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_Jrec'], size = (self.N, self.N))
        # self.J = np.zeros ((self.N, self.N))

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))

        # This is the hint signal
        try:
            self.Hint = par['hint_shape']
            self.Jhint = np.random.normal (0., par['sigma_hint'], size = (self.N, self.H))
        except KeyError:
            self.Hint = 0
            self.Jhint = None

        self.Jout = np.random.normal (0., par['sigma_Jout'], size = (self.O, self.N))
        # self.Jout = np.zeros ((self.O, self.N))

        # Remove self-connections
        np.fill_diagonal (self.J, 0.)

        # Impose reset after spike
        self.s_inh = -par['s_inh']
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h']

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']
        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros (self.N)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.policy_thr_tau = par['policy_thr_tau'] if 'policy_thr_tau' in par else 20

        # Check whether output should be put through a sigmoidal gate
        self.outsig = par['outsig'] if 'outsig' in par else False

        # Save the way policy should step in closed-loop scenario
        self.step_mode = 'amax'#par['step_mode'] if 'step_mode' in par else 'UNSET'

        # Here we save the params dictionary
        self.par = par

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))

        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))


    def step (self, inp):
        itau_m = self.itau_m
        itau_s = self.itau_s

        self.S_hat = self.S_hat * itau_s + self.S [:] * (1. - itau_s)

        self.H = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + self.Jin @ inp + self.h)\
                                                             + self.Jreset @ self.S
        #
        self.lam = self._sigm ( self.H, dv = self.dv )
        self.dH = self.dH  * itau_m + self.S_hat * (1. - itau_m)

        self.S =  np.heaviside(self.lam - np.random.rand(self.N,),0)

        # Here we return the chosen next action
        action, out = self.policy (self.S)

        return action, out

    def step_det (self, inp):
        itau_m = self.itau_m
        itau_s = self.itau_s

        self.S_hat = self.S_hat * itau_s + self.S * (1. - itau_s)

        self.H = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + self.Jin @ inp + self.h)\
                                                             + self.Jreset @ self.S

        self.lam = self._sigm ( self.H, dv = self.dv )
        self.dH = self.dH  * itau_m + self.S_hat * (1. - itau_m)

        self.S = self._sigm (self.H, dv = self.dv) - 0.5 > 0.

        # Here we return the chosen next action
        action, out = self.policy (self.S)
        self.out = out
        self.action = action

        return action, out

    def learn (self, r):

        alpha_J = 0.01

        dJ = np.outer ((self.S - self.lam), self.dH)
        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ

        self.J += 0.1*r*self.dJfilt
        self.J -= self.J*0.000

    def learn_error (self, r):

        ch = 2.

        alpha_J = 0.002
        ac_vector = np.zeros((3,))
        ac_vector[self.action] = 1
        dJ = np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)

        self.entropy = - np.sum(self.prob*np.log(self.prob))

        dJ_ent = - np.outer (self.Jout.T@(self.prob*np.log(self.prob ) + self.entropy)*self._dsigm (self.H, dv = 1.), self.dH)
        dJ_ent_out = - np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)

        dJ_out =  np.outer((ac_vector - self.out), self.state_out.T)
        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out

        self.dJ_aggregate += (r*self.dJfilt + ch*dJ_ent*0)
        self.dJout_aggregate += (r*self.dJfilt_out + ch*dJ_ent_out*0)

    def update_J (self, r):

        self.J = self.adam_rec.step (self.J, self.dJ_aggregate)#
        np.fill_diagonal (self.J, 0.);

        self.Jout = self.adam_out.step (self.Jout, self.dJout_aggregate)
        self.Jout = self.Jout + self.par["alpha_rout"]*self.dJout_aggregate

        self.dJ_aggregate=0
        self.dJout_aggregate=0

    def learn_model(self,s_pred,r_pred,state,r):

         self.dJout_s_aggregate += np.outer((state-s_pred),self.state_out )# 0.00001
         self.dJout_r_aggregate += np.outer((r-r_pred),self.state_out )

         dJ_s = np.outer (self.Jout_s_pred.T@(state-s_pred)*self._dsigm (self.H, dv = 1.), self.dH)
         dJ_r = np.outer (self.Jout_r_pred.T@(r-r_pred)*self._dsigm (self.H, dv = 1.), self.dH)

         eta_rec_r = 0.5

         self.dJ_aggregate += dJ_s
         self.dJ_aggregate += dJ_r*eta_rec_r

    def model_update( self ):

        self.J = self.adam_rec.step (self.J, self.dJ_aggregate)
        self.Jout_s_pred =  self.adam_out_s.step (self.Jout_s_pred, self.dJout_s_aggregate) #
        self.Jout_r_pred =  self.adam_out_r.step (self.Jout_r_pred, self.dJout_r_aggregate) #

        self.dJout_r_aggregate=0
        self.dJout_s_aggregate=0
        self.dJ_aggregate=0

    def policy (self, state, mode = 'amax', explore = False):
        self.state_out = self.state_out * self.itau_ro  + state * (1 - self.itau_ro)

        out = self.Jout @ self.state_out*10.*.5
        if self.outsig:
            #out = self._sigm(out,0.1)+0.00001
            out = np.exp(out) / np.sum(np.exp(out))
        prob = out#p.exp(out) / np.sum(np.exp(out))
        self.prob = prob

        action = np.random.choice(len(out), p = prob)
        #action = random.choices( population=[0, 1, 2],weights=out,k=1)

        return int(action),out

    def prediction (self):

        s_pred = self.Jout_s_pred @ self.state_out
        r_pred = self.Jout_r_pred @ self.state_out
        return s_pred,r_pred


    def reset (self, init = None):
        self.S [:] = init if init is not None else np.zeros (self.N)
        self.S_hat [:] = self.S [:] * self.itau_s if init is not None else np.zeros (self.N)

        self.state_out [:] *= 0

        self.H [:] = self.Vo

    def forget (self, J = None, Jout = None):
        self.J    = np.random.normal (0., self.par['sigma_Jrec'], size = (self.N, self.N)) if J    is None else J.copy()
        self.Jout = np.random.normal (0., self.par['sigma_Jout'], size = (self.O, self.N)) if Jout is None else Jout.copy()

    def save (self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jin, self.Jteach, self.Jout, self.J, self.par)

        np.save (filename, np.array (data_bundle, dtype = np.object))

    @classmethod
    def load (cls, filename):
        data_bundle = np.load (filename, allow_pickle = True)

        Jin, Jteach, Jout, J, par = data_bundle

        obj = AGEMO (par)

        obj.Jin = Jin.copy ()
        obj.Jteach = Jteach.copy ()
        obj.Jout = Jout.copy ()
        obj.J = J.copy ()

        return obj
