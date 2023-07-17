import warnings
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

from typing import Tuple, Dict, Any, Callable, Optional, Union

black = (0, 0, 0) 
white = (255, 255, 255)
grey  = (30, 30, 30)

def default(var, val):
    return val if var is None else var

class VanillaButtonFood:
    '''
        Button Food environment where an agent explore a 2D (continuous) world
        to reach a target location where food is stored. The food is initially
        locked and the agent must first press a button to unlock the target.
    '''

    metadata = {
        'render_modes' : ['human', 'rgb_array'],
        'font_size' : 30,
        'render_fps'   : 24,
        'window_size'  : (512, 512),
    }

    def __init__(
        self,
        init_agent  : Optional[np.ndarray] = None,
        init_target : Optional[np.ndarray] = None,
        init_button : Optional[np.ndarray] = None,
        domain : Tuple[int, int] = (0, 1),
        render_mode : Optional[str] = None,
        input_dim : int = 25,
        time_limit : int = 500,
        max_speed : float = .01,
        target_radius : float = 0.025,
        button_radius : float = 0.040,
        num_actions : int = 8,
    ) -> None:
        super(VanillaButtonFood, self).__init__() 

        self._input_dim = input_dim 
        self._button_radius = button_radius
        self._target_radius = target_radius 

        # The action space corresponds to the instantaneous speed in the 2D world
        # along the x- and y-direction.
        self.max_speed = max_speed
        self.discrete_action_space = num_actions > 0
        self.num_actions = num_actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Here we collect environment duration and span
        self.time_limit = time_limit
        self.domain = np.array(domain)

        # Internal timekeeping & done flag
        self.time = 0 
        self.done = False 
        self.reward = 0

        # Here we init the position of target and agent and button
        self._agent_location  = default(init_agent, np.ones(2) * 0.5)
        self._target_location = default(init_target, np.random.random(2)) 
        self._button_location = default(init_button, np.random.random(2))
        self._agent_velocity  = np.zeros(2) 

        self._target_identity = np.random.randint(low=1, high=45)

        # Here we initialize the flag the signal whether the button was pressed
        self.is_button_pressed = False

        # Callback for recording environment quantities
        self.step_callback = None

        # Variable needed for rendering
        self.clock = None
        self.window = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            # The environment observation is the gaussian-encoded position
            # difference from target and button concatenated together
            'agent_target' : self.encode(self._target_location - self._agent_location, dim = self._input_dim, domain = self.domain),
            'agent_button' : self.encode(self._button_location - self._agent_location, dim = self._input_dim, domain = self.domain),
            'agent' : self._agent_location,
            'target': self._target_location,
            'button': self._button_location,
        }
    
    def _get_info(self) -> Dict[str, np.ndarray]:
        return {
            'env' : 'button-food',
            'done' : self.done,
            'time' : self.time,
            'reward' : self.reward,
            'target_distance' : euclidean(self._agent_location, self._target_location),
            'button_distance' : euclidean(self._agent_location, self._button_location),
            'button_pressed'  : self.is_button_pressed,
        }

    def step(self, action : Union[np.ndarray, Tuple[int, int]], **kwd):
        if self.discrete_action_space:
            theta = np.linspace(0, 2 * np.pi, num = self.num_actions, endpoint = False)
            self._agent_velocity = np.array(
                                    [np.cos(theta[action]), np.sin(theta[action])]
                                ) * self.max_speed
        else:
            self._agent_velocity = np.array(action).clip(-self.max_speed, self.max_speed)

        # Action is the instantaneous speed in the x- and y-direction
        self._agent_location += self._agent_velocity
        self._agent_location = np.clip(
            self._agent_location, *self.domain,
        )

        # * Update the environment flag and compute the reward
        # Compute the distance to target and button
        target_dist = euclidean(self._agent_location, self._target_location)
        button_dist = euclidean(self._agent_location, self._button_location)

        self.is_button_pressed |=  button_dist < self._button_radius 
        self.done |= target_dist < self._target_radius and self.is_button_pressed or self.time > self.time_limit

        # Compute the reward
        # reward = self.is_button_pressed / target_dist

        # Reward is the alignment of the agent velocity with the
        # target position vector with respect to the agent position
        self.reward = 1 - cosine(self._agent_velocity, self._target_location - self._agent_location)

        # Timekeeping for out-of-time-limit control
        self.time += 1
        self.truncated = self.time > self.time_limit

        self.obs  = self._get_obs()
        self.info = self._get_info()

        if self.step_callback:
            self.step_callback(self, **kwd)

        return self.obs, self.reward, self.done, self.truncated, self.info 

    def reset(
        self,
        seed : int = None,
        init_agent  : Optional[np.ndarray] = None,
        init_target : Optional[np.ndarray] = None,
        init_button : Optional[np.ndarray] = None,
        options : Any = None,
    ):
        options = default(options, {'button_pressed' : False})

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._agent_location  = np.array(default(init_agent,  np.random.uniform(*self.domain, size=2)))
        self._button_location = np.array(default(init_button, np.random.uniform(*self.domain, size=2)))
        self._target_location = np.array(default(init_target, self._button_location))

        self._target_identity = np.random.randint(0, 40, size=(2,))

        while euclidean(self._button_location, self._target_location) < self._button_radius or\
              euclidean(self._agent_location,  self._target_location) < 0.2:
            self._target_location = np.random.uniform(*self.domain, size=2)

        self.is_button_pressed = options['button_pressed'] 
        self.time = 0
        self.done = False

        obs  = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def _agent_heading(self):
        vx, vy = self._agent_velocity

        if vy > vx and vy <= -vx: return 0 # LEFT
        if vy > vx and vy >= -vx: return 1 # DOWN
        if vy < vx and vy >= -vx: return 2 # RIGHT
        if vy < vx and vy <= -vx: return 7 # UP

        return 0

    def register_step_callback(self, fn : Callable) -> None:
        self.step_callback = fn

    def unregister_step_callback(self) -> None:
        self.step_callback = None 

    @classmethod
    def encode(
        cls,
        pos : Union[np.ndarray, Tuple[int, int]],
        dim : int = 25,
        std : float = 0.1,
        domain : Tuple[float, float] = (0, 1),
    ) -> np.ndarray:
        
        if std * dim < 1:
            warnings.warn('Standard deviation used for encoding might be too small')
        
        pos = np.array(pos)
        
        x, y = pos 
        l, r = domain
        b, t = domain

        w = r - l
        h = t - b

        x = np.clip(x, l, l + w)
        y = np.clip(y, b, b + h)

        mux = np.linspace(l, l + w, dim) - .5 * w / dim
        muy = np.linspace(b, b + h, dim) - .5 * h / dim

        codex = np.exp(-0.5 * ((x - mux) / (w * std))**2)
        codey = np.exp(-0.5 * ((y - muy) / (h * std))**2)

        return np.concatenate([codex, codey], axis = -1)
    
    @classmethod
    def decode(
        cls,
        encoded_state : np.ndarray,
        dim : int = 25
    ) -> np.ndarray:
        x_state, y_state = encoded_state[..., :dim], encoded_state[..., dim:]

        x_idxs = np.argsort(x_state, axis=-1)[..., :]
        y_idxs = np.argsort(y_state, axis=-1)[..., :]
        
        x_weight = np.take_along_axis(x_state, x_idxs, axis = -1)
        y_weight = np.take_along_axis(y_state, y_idxs, axis = -1)

        x_pos = np.average(x_idxs / dim, weights = x_weight + 1e-10, axis = -1)
        y_pos = np.average(y_idxs / dim, weights = y_weight + 1e-10, axis = -1)

        return np.stack((x_pos, y_pos), axis = 0)
    
    @classmethod
    def build_expert(
        cls,
        init   : Optional[Tuple[int, int]] = None,
        food   : Optional[Tuple[int, int]] = None,
        button : Optional[Tuple[int, int]] = None,
        steps : Tuple[int, int] = (80, 80),
        offs : Tuple[int, int] = (0, 0),
        dim : int = 25,
        domain : Tuple[float, float, float, float] = (0, 0, 1, 1),
        norm : bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
            Compute the expert trajectories that would solve the environment. The
            agent would move into a straight line towards the button and then, from
            there, straight towards the target.
        '''

        init = default(init, (0.5, 0.5))
        food = default(food,     np.random.random(2))
        button = default(button, np.random.random(2))

        s1, s2 = steps
        off1, off2 = offs

        # Move from init to button in `s1` steps with first offset
        ix, iy = init
        fx, fy = food
        bx, by = button

        off1 = np.array([*[0] * off1])
        off2 = np.array([*[0] * off2])

        px1 = np.concatenate([off1, np.linspace(ix, bx, s1)])
        py1 = np.concatenate([off1, np.linspace(iy, by, s1)])

        vx1 = np.array([*off1, *[(bx - ix) / s1] * s1])
        vy1 = np.array([*off1, *[(by - iy) / s1] * s1])

        # Move from button to target into `s2` steps with second offset
        px2 = np.concatenate([off2, np.linspace(bx, fx, s2)])
        py2 = np.concatenate([off2, np.linspace(by, fy, s2)])

        vx2 = np.array([*off2, *[(fx - bx) / s2] * s2])
        vy2 = np.array([*off2, *[(fy - by) / s2] * s2])

        px, py = np.concatenate([px1, px2]), np.concatenate([py1, py2])
        vx, vy = np.concatenate([vx1, vx2]), np.concatenate([vy1, vy2])

        p = np.stack([px, py])
        v = np.stack([vx, vy], dim = -1)[:, None]

        p = cls.encode(p, dim, domain)

        if norm: v /= np.amax(np.abs(v), dim = (0, 1), keepdim = True)

        # Returned tensor have shape [time, batch, dim]
        return p, v