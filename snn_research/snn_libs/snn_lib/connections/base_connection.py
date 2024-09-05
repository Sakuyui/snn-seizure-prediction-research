class AbstractConnection(object):
    def __init__(self, output_index = 0):
        self._cached_states = None
        self._states = None
        self._output_index = output_index
    
    def do_update_states(self):
        if (self._cached_states == None):
            raise ValueError("Cache is empty.")
        self._states = self._cached_states
        self._cached_states = None
        
    @property
    def states(self):
        return self._states
    
    @property
    def cached_states(self):
        return self._cached_states
    
    def initialize(self, maintain_weights = False):
        raise NotImplemented
    
    def pseudo_update_states(self, u = None):
        raise NotImplemented
    
    def get_output(self, u = None):
        if self.cached_states != None:
            return self.cached_states[self._output_index]
        else:
            return self.states[self._output_index]
    
    def cache_states(self, states):
        self._cached_states = states
        
    def __call__(self, u):
        if self.states == None:
            raise ValueError("States have not been initialized yet.")
        if(self.cached_states != None):
            raise RuntimeError("cached output should write back to `states` property by `do_update_states()` before further update.")
        connection_states = self.pseudo_update_states(u)
        connection_output = self.get_output(u)
        return connection_states, connection_output
        