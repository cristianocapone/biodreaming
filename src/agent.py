"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Towards biologically plausible Dreaming and Planning in recurrent spiking networks
"""
import pickle
import numpy as np
from abc import abstractclassmethod

from .utils import default
from .optimizer import Adam

from typing import Tuple, Dict, Any, Callable

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

class AGEMONE:

    def __init__ (self, config : Dict[str, Any]):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O = config['N'], config['I'], config['O']

        self.dt = config['dt']

        self.itau_m    = np.exp(-self.dt / config['tau_m'])
        self.itau_s    = np.exp(-self.dt / config['tau_s'])
        self.itau_ro   = np.exp(-self.dt / config['tau_ro'])
        self.itau_star = np.exp(-self.dt / config['tau_star'])

        self.dv = config['dv']

        self.hidden = config['hidden_steps'] if 'hidden_steps' in config else 1

        # This is the network connectivity matrix
        self.J_rec = np.random.normal(0., config['sigma_Jrec'], size = (self.N, self.N))
        self.J_rec /= np.sqrt(self.N) # Normalize the connectivity matrix

        # Remove self-connections from synaptic matrix
        np.fill_diagonal(self.J_rec, 0.)

        # Impose reset after spike
        self.s_inh = -config['s_inh']
        self.J_reset = np.diag(np.ones(self.N) * self.s_inh)

        # External constant field and initial membrane potential
        self.h  = config['h']
        self.Vo = config['Vo']

        # Recurrent matrix optimizer
        self.adam_rec = Adam(alpha=config['alpha_rec'])

        # Here we save the model configuration
        self.config = config

        # Callback for arbitrary quantity recording from agents
        self.step_callback = None

        # Reset takes care of initializing the membrane potential and spikes
        # and all filtered accumulators
        self.reset()

    def _sigm (self, x : np.ndarray, dv : float = None) -> np.ndarray:
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0

        # Here we apply numerically stable version of sigmoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))

        return out

    def _dsigm (self, x : np.ndarray, dv : float = None) -> np.ndarray:
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))

    def step(self, state : np.ndarray, action : np.ndarray | None = None, deterministic : bool = False, **kwd) -> Any:
        # Collect the state and action variables for monitoring
        self.inp_state  = state
        self.inp_action = action
        
        inp = state if action is None else np.concatenate((state, action))
        
        itau_m = self.itau_m
        itau_s = self.itau_s

        self.S_hat = self.S_hat * itau_s + self.S * (1. - itau_s)

        self.H = self.H * itau_m + (1. - itau_m) * (
                                                    self.J_rec @ self.S_hat +\
                                                    self.J_input @ inp + self.h
                                                ) +\
                                self.J_reset @ self.S
        
        self.lam = self._sigm( self.H, dv = self.dv)
        self.dH = self.dH  * itau_m + self.S_hat * (1. - itau_m)

        self.S = np.heaviside(self.lam - (0.5 if deterministic else np.random.rand(self.N)), 0)

        # Here we return the chosen next action
        out = self.policy(self.S)

        if self.step_callback:
            self.step_callback(self, **kwd)

        return out

    def register_step_callback(self, fn : Callable) -> None:
        self.step_callback = fn

    def unregister_step_callback(self) -> None:
        self.step_callback = None 

    @abstractclassmethod
    def policy(self, spikes : np.ndarray, *args, **kwargs) -> Any:
        pass

    def reset(self, spikes : np.ndarray | None = None):
        self.S     = default(spikes, np.zeros(self.N))
        self.S_hat = default(spikes, np.zeros(self.N)) * self.itau_s

        self.state_out = np.zeros(self.N)

        self.H  = np.ones (self.N) * self.Vo
        self.dH = np.zeros(self.N)

    def forget(self, J_rec : np.ndarray | None = None, J_out  : np.ndarray | None = None):
        self.J_rec = np.random.normal(0., self.config['sigma_Jrec'], size = (self.N, self.N)) if J_rec is None else J_rec.copy()
        self.J_out = np.random.normal(0., self.config['sigma_Jout'], size = (self.O, self.N)) if J_out is None else J_out.copy()

    def save (self, filename : str):
        # Here we collect the relevant quantities to store
        bundle = {name : value for name, value in self.parameters.items()}

        with open(filename, 'wb') as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, filename : str):
        with open(filename, 'rb') as f:
            bundle = pickle.load(f)

        obj = cls(bundle['config'])

        for name, value in bundle:
            setattr(obj, name, value)

        return obj

class Actor(AGEMONE):

    def __init__(self, config : Dict[str, Any]) -> None:
        # Agent has a single J_out for action prediction
        self.J_out = np.random.normal(0., config['sigma_Jout'], size = (config['O'], config['N']))
 
        self.adam_out = Adam(alpha=config['alpha_out'])

        # This is the network input and teach matrices
        self.J_input = np.random.normal(0., config['sigma_input'], size = (config['N'], config['I']))
        self.J_teach = np.random.normal(0., config['sigma_teach'], size = (config['N'], config['O']))

        super().__init__(config)

        # Configure Actor parameters
        self.parameters = {
            'J_input' : self.J_input,
            'J_teach' : self.J_teach,
            'J_out' : self.J_out,
            'J_rec' : self.J_rec,
            'config' : self.config,
        }

    def policy(self, spikes : np.ndarray, mode : str | None = None):
        mode = default(mode, self.config['step_mode'])

        self.state_out = self.state_out * self.itau_ro  + spikes * (1 - self.itau_ro)

        p_out = self.J_out @ self.state_out
        if self.config['outsig']: p_out = np.exp(p_out) / np.sum(np.exp(p_out))
        
        # match mode:
        #     case 'amax': action = np.argmax(p_out)
        #     case 'prob': action = int(np.random.choice(len(p_out), p = p_out))
        #     case 'raw' : action = p_out
        if   mode == 'amax': action = np.argmax(p_out)
        elif mode == 'prob': action = int(np.random.choice(len(p_out), p = p_out))
        elif mode == 'raw' : action = p_out
        else: raise ValueError(f'Unknown policy mode: {mode}')
        
        self.out_prob   = p_out
        self.out_action = action

        return action

    def accumulate_evidence(self, reward : float, alpha_J : float = 0.01):
        # From last action (int) compute the action vector and its difference
        # with the probability vector
        act_vec = np.zeros(self.O)
        act_vec[self.out_action] = 1
        act_diff = act_vec - self.out_prob

        pseudo = self._dsigm(self.H, dv = 1.)
        
        # Compute the synaptic matrix gradient for recurrent and output connections
        dJ_rec = np.outer(self.J_out.T @ act_diff * pseudo, self.dH)
        dJ_out = np.outer(act_diff, self.state_out.T)

        # Accumulate the gradient evidence
        self.dJ_filt_rec = self.dJ_filt_rec * (1 - alpha_J) + dJ_rec
        self.dJ_filt_out = self.dJ_filt_out * (1 - alpha_J) + dJ_out

        self.dJ_rec_accumulate += reward * self.dJ_filt_rec
        self.dJ_out_accumulate += reward * self.dJ_filt_out

    def learn_from_evidence(self):
        self.J_rec = self.adam_rec.step(self.J_rec, self.dJ_rec_accumulate)
        self.J_out = self.adam_out.step(self.J_out, self.dJ_out_accumulate)

        np.fill_diagonal(self.J_rec, 0)
        self.dJ_rec_accumulate = 0
        self.dJ_out_accumulate = 0

    def reset(self, spikes : np.ndarray | None = None):
        super().reset(spikes)

        self.dJ_filt_rec = 0
        self.dJ_filt_out = 0

        self.dJ_rec_accumulate = 0
        self.dJ_out_accumulate = 0

class Planner(AGEMONE):

    def __init__(self, config : Dict[str, Any]) -> None:
        # Planner needs additional parameters for state prediction
        self.J_out_s = np.zeros((config['I'], config['N']))
        self.J_out_r = np.zeros((1, config['N']))

        # Each output matrix has its own optimizer
        self.adam_out_s = Adam(alpha=config['alpha_out'])
        self.adam_out_r = Adam(alpha=config['alpha_out'])

        # This is the network input and teach matrices
        self.J_input = np.random.normal(0., config['sigma_input'], size = (config['N'], config['I'] + config['O']))

        super().__init__(config)

        # Configure Actor parameters
        self.parameters = {
            'J_input' : self.J_input,
            'J_out_s' : self.J_out_s,
            'J_out_r' : self.J_out_r,
            'J_rec'  : self.J_rec,
            'config' : self.config,
        }

    def policy(self, spikes : np.ndarray):
        self.state_out = self.state_out * self.itau_ro  + spikes * (1 - self.itau_ro)
 
        self.next_state  = self.J_out_s @ self.state_out
        self.next_reward = self.J_out_r @ self.state_out

        return self.next_state, self.next_reward

    def accumulate_evidence(self, pred : Tuple[np.ndarray, float], targ  : Tuple[np.ndarray, float], eta_rec_r : float = 0.5):
        s_pred, r_pred = pred
        s_targ, r_targ = targ

        self.dJ_out_s_accumulate += np.outer(s_targ - s_pred, self.state_out)
        self.dJ_out_r_accumulate += np.outer(r_targ - r_pred, self.state_out)

        # dJ_s = np.outer(self.J_out_s.T @ (s_targ - s_pred) * self._dsigm(self.H, dv = 1.), self.dH)
        # dJ_r = np.outer(self.J_out_r.T @ (r_targ - r_pred) * self._dsigm(self.H, dv = 1.), self.dH)

        # self.dJ_rec_accumulate += dJ_s
        # self.dJ_rec_accumulate += dJ_r * eta_rec_r

    def learn_from_evidence(self) -> None: 
        # self.J_rec = self.adam_rec.step(self.J_rec, self.dJ_rec_accumulate)
        self.J_out_s = self.adam_out_s.step(self.J_out_s, self.dJ_out_s_accumulate)
        self.J_out_r = self.adam_out_r.step(self.J_out_r, self.dJ_out_r_accumulate)

        # np.fill_diagonal(self.J_rec, 0)

        self.dJ_out_r_accumulate = 0
        self.dJ_out_s_accumulate = 0
        self.dJ_rec_accumulate = 0

    def reset(self, spikes : np.ndarray | None = None) -> None:
        super().reset(spikes)

        self.dJ_rec_accumulate = 0
        self.dJ_out_s_accumulate = 0
        self.dJ_out_r_accumulate = 0

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

        # Here we apply numerically stable version of sigmoid activation
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
        # ! WHY PUT THIS NUMBER HERE? 
        alpha_J = 0.01

        dJ = np.outer ((self.S - self.lam), self.dH)

        # ! SHOUDN'T IT BE alpha_J * dJ??
        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ

        # ! WHY MULTIPLY BY ZERO? WHY THE 0.1 BEFORE r?
        self.J += 0.1*r*self.dJfilt
        self.J -= self.J*0.000

    def learn_error (self, r):
        # ! WHAT IS THIS CH?
        ch = 2

        alpha_J = 0.002
        ac_vector = np.zeros((3,))
        ac_vector[self.action] = 1
        dJ = np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)

        self.entropy = - np.sum(self.prob*np.log(self.prob))

        dJ_ent = - np.outer (self.Jout.T@( self.prob*np.log(self.prob ) + self.entropy)*self._dsigm (self.H, dv = 1.), self.dH)
        dJ_ent_out = - np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)

        dJ_out =  np.outer((ac_vector - self.out), self.state_out.T)
        self.dJfilt = self.dJfilt*(1-alpha_J) + dJ
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out

        # ! SHADY MULTIPLY BY ZERO
        self.dJ_aggregate += (r*self.dJfilt + ch*dJ_ent*0)
        self.dJout_aggregate += (r*self.dJfilt_out + ch*dJ_ent_out*0)

    def update_J(self, r):

        self.J = self.adam_rec.step(self.J, self.dJ_aggregate)
        np.fill_diagonal (self.J, 0.)

        # ! WHY TWO UPDATE OF J_OUT??
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

        # ! WHY THE * 10. * .5??
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
