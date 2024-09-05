class AbstractNeuron(object):
    def __init__(self, hyperparameters = None, states = None, output_index = 0):
        self._hyperparameters = hyperparameters
        self._states = states
        self._output_index = 0
        self._cached_states = None
        self.neuron_count = 1
        
    @property
    def n_neuron(self):
        return self.neuron_count
    
    @property
    def states(self):
        return self._states

    @property
    def hyperparameters(self):
        return self._hyperparameters
    
    @property
    def cached_states(self):
        return self._cached_states
    
    def cache_states(self, states):
        self._cached_states = states
        
    def set_hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters

    def reset_states(self):
        raise NotImplementedError
    
    def pseudo_update_states(self, u):
        raise NotImplementedError
    
    def reset_time(self, t = 0):
        self.t = t
    
    def do_update_states(self):
        if self.cached_states == None:
            raise ValueError("No updated states are stored in cache.")
        self._states = self.cached_states
        self._cached_states = None
    
    def initialize(self):
        raise NotImplementedError
    
    def get_output(self, u = None):
        if self.cached_states != None:
            return self.cached_states[self._output_index]
        else:
            return self.states[self._output_index]
    
    def __call__(self, u):
        if self.states == None:
            raise ValueError("States have not been initialized yet.")
        if(self.cached_states != None):
            raise RuntimeError("cached output should write back to `states` property by `do_update_states()` before further update.")
        neuron_states = self.pseudo_update_states(u)
        neuron_output = self.get_output(u)
        
        return neuron_states, neuron_output
        