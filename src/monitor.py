
import pickle
from typing import Tuple, Any
from collections import defaultdict

class Recorder:
    '''
        Monitor class for recording various attributes from both the
        agents and environment variables.    
    '''

    def __init__(self, targets : Tuple[str, ...]) -> None:
        self.names = targets
        self.vault = defaultdict(list)
        self.buffer = defaultdict(list)

        self.criterion = lambda : True

    def __call__(self, obj: Any, **kwd) -> None:
        if not self.criterion(**kwd):
            return None
        
        for name in self.names:
            value = getattr(obj, name)
            self.buffer[name].append(value)

        return None
    
    def __setitem__(self, key : str, value : Any) -> None:
        self.vault[key].append(value)

    def __getitem__(self, key: str) -> list:
        return self.vault[key]
    
    def commit_buffer(self) -> None:
        for name in self.names:
            if len(self.buffer[name]): self.vault[name].append(self.buffer[name])

        self.buffer = defaultdict(list)

    def reset(self) -> None:
        self.vault = defaultdict(list)
        self.buffer = defaultdict(list)

    def dump(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.vault, f)