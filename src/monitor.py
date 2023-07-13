
import pickle
from typing import Tuple, Any
from collections import defaultdict

class Recorder:
    '''
        Monitor class for recording various attributes from both the
        agents and environment variables.    
    '''

    def __init__(self, targets : Tuple[str, ...], do_raise : bool = False) -> None:
        self.names = targets
        self.vault = defaultdict(list)
        self.buffer = defaultdict(list)

        self.criterion = lambda : True

        self.do_raise = do_raise

    def __call__(self, obj: Any, **kwd) -> None:
        if not self.criterion(**kwd):
            return None
        
        for name in self.names:
            try:
                obj_name, attr_name = name.split('.')

                if obj_name.lower() not in str(obj).lower(): continue
            except ValueError:
                attr_name = name

            try:
                attr_name, attr_key = attr_name.split(':')
            except (ValueError):
                attr_key = None

            try:
                value = getattr(obj, attr_name)
                value = value if attr_key is None else value[attr_key]
                self.buffer[name].append(value)

            # Just skip whether environment or agent does not have the
            # chosen attribute (re-raise if )
            except AttributeError as E:
                if self.do_raise: raise E
                else: pass

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