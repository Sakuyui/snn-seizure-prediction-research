from snn_lib.neuron_models.base_neuron_model import AbstractNeuron
import numpy as np

class CurrentBasedLIFNeuron(AbstractNeuron):
    def __init__(self, N, hyperparameters):
        super().__init__()
        
        ## Hyperparameters
        self.set_hyperparameters(
            {
                'dt': hyperparameters.get('dt', 1e-4), 
                'tref': hyperparameters.get('tref', 5e-3), 
                'tc_m': hyperparameters.get('tc_m', 1e-2), 
                'vrest': hyperparameters.get('vrest', -60), 
                'vreset': hyperparameters.get('vreset', -60), 
                'vthr': hyperparameters.get('vthr', -50), 
                'vpeak': hyperparameters.get('vpeak', 20)
            }
        )

        ## parameters
        self.neuron_count = N
        self.INDEX_FIRED = 0
        self.INDEX_V = 1
        self.INDEX_TLAST = 2
        self.INDEX_TCOUNT = 3
        self.INDEX_LAST_FIRED_V = 4

        self.initialize()        
        

       
    
    @property
    def v_(self):
        return self.states[self.INDEX_LAST_FIRED_V]

    @property
    def tref(self):
        return self._hyperparameters['tref']

    @property
    def tc_m(self):
        return self._hyperparameters['tc_m']

    @property
    def vrest(self):
        return self._hyperparameters['vrest']
    
    @property
    def vreset(self):
        return self._hyperparameters['vreset']

    @property
    def vthr(self):
        return self._hyperparameters['vthr']

    @property
    def vpeak(self):
        return self._hyperparameters['vpeak']
    
    @property
    def dt(self):
        return self._hyperparameters['dt']
    
    def pseudo_update_states(self, u):
        states = self.states
        
        v = states[self.INDEX_V]
        tlast = states[self.INDEX_TLAST]
        tcount = states[self.INDEX_TCOUNT]
        last_fired_v = states[self.INDEX_LAST_FIRED_V]
        
        dv = (self.vrest - v + u) / self.tc_m
        v = v + (self.dt * tcount > (tlast + self.tref)) * dv * self.dt

        s = 1 * (v >= self.vthr)


        tlast = tlast * (1 - s) + self.dt * tcount * s
        v = v * (1 - s) + self.vpeak * s
        
        last_fired_v = v
        
        v = v * (1 - s) + self.vreset * s
        tcount += 1
        self.cache_states([s, v, tlast, tcount, last_fired_v])
        return self.cached_states
 
    def initialize(self, random_state = False):
        if self.states == None:
            self._states = [None, None, None, None, None]

        states = self.states
        if random_state:
            states[self.INDEX_V] = self.vreset + np.random.rand(self.n_neuron) * (self.vthr - self.vreset)
        else:
            states[self.INDEX_V] = self.vreset * np.ones(self.n_neuron)
            states[self.INDEX_TLAST] = 0
            states[self.INDEX_TCOUNT] = 0      
        states[self.INDEX_FIRED] = np.zeros(self.n_neuron).astype(int)
        states[self.INDEX_LAST_FIRED_V] = 0
        self._states = states
        self._cached_states = None
        
class ConductanceBasedLIFNeuron(AbstractNeuron):
    def __init__(self, N, hyperparameters):
        super().__init__()
        self.neuron_count = N
        self.set_hyperparameters(
            {
                'dt': hyperparameters.get("dt", 1e-4),
                'tref':hyperparameters.get("tref", 5e-3), 
                'tc_m': hyperparameters.get("tc_m", 1e-2), 
                'vrest': hyperparameters.get("vret", -60),
                'vreset': hyperparameters.get("vreset", -60), 
                'vthr': hyperparameters.get("vthr", -50), 
                'vpeak': hyperparameters.get("vpeak", 20), 
                'e_exc': hyperparameters.get("e_exc", 0), 
                'e_inh': hyperparameters.get("e_inh", -100)
            }
        )
        
        self.initialize()
        self.do_update_states()
        self.INDEX_FIRED_OUTPUT = 0
        self.INDEX_V = 1
        self.INDEX_LAST_FIRED_V = 2
        self.INDEX_TLAST = 3
        self.INDEX_TCOUNT = 4

    @property
    def dt(self):
        return self.hyperparameters['dt']
    
    @property
    def tref(self):
        return self.hyperparameters['tref']

    @property
    def tc_m(self):
        return self.hyperparameters['tc_m']

    @property
    def vrest(self):
        return self.hyperparameters['vrest']

    @property
    def vreset(self):
        return self.hyperparameters['vrest']

    @property
    def vthr(self):
        return self.hyperparameters['vthr']

    @property
    def vpeak(self):
        return self.hyperparameters['vpeak']

    @property
    def e_exc(self):
        return self.hyperparameters['e_exc']
    
    @property
    def e_inh(self):
        return self.hyperparameters['e_inh']
        
    def pseudo_update_states(self, u):
        g_exc, g_inh = u[0], u [1]
        
        states = self.states
        v = states[self.INDEX_V]
        tcount = states[self.INDEX_TCOUNT]
        tlast = self.INDEX_TLAST
        
        I_synExc = g_exc * (self.e_exc - v) # 兴奋性输入
        I_synInh = g_inh * (self.e_inh - v) # 抑制性输入
        
        dv = (self.vrest - v + I_synExc + I_synInh) / self.tc_m
        v = v + ((self.dt * tcount) > (tlast + self.tref)) * dv * self.dt
                
        s = 1 * (v >= self.vthr)
                
        tlast = tlast * (1 - s) + self.dt * tcount * s # 発火時刻更新
        v = v * (1 - s) + self.vpeak * s # 閾値を超えると膜電位を vpeak にする
        v_ = v 
        v = v * (1 - s) + self.vreset * s
        tcount += 1
        
        self.cache_states([s, v, v_, tlast, tcount])
        return self.cached_states
                
    
    def initialize(self, random_state=False, maintain_weights = False):
        if self.states == None:
            self._states = [None, None, None, None, None]
        states = self.states
        if random_state:
            states[self.INDEX_V] = self.vreset + np.random.rand(self.n_neuron) * (self.states[self.vthr] - self.vreset)
        else:
            states[self.INDEX_V] = self.vreset * np.ones(self.n_neuron)
        states[self.INDEX_TLAST] = 0
        states[self.INDEX_TCOUNT] = 0
        states[self.INDEX_FIRED_OUTPUT] = 0
        states[self.INDEX_LAST_FIRED_V] = 0
        self._states = states
        self._cached_states = None


class DiehlAndCook2015LIFNeuron(AbstractNeuron):
    def __init__(self, N, hyperparameters):         
        super().__init__()
        self.neuron_count = N
        self.set_hyperparameters({
            'dt': hyperparameters.get('dt', 1e-3), 
            'tref': hyperparameters.get('tref', 5e-3), 
            'tc_m': hyperparameters.get('tc_m', 1e-1), 
            'vrest': hyperparameters.get('vrest', -65), 
            'vreset': hyperparameters.get('vreset', -65), 
            'init_vthr': hyperparameters.get('init_vthr', -52), 
            'vpeak': hyperparameters.get('vpeak', 20), 
            'theta_plus': hyperparameters.get('theta_plus', 0.05), 
            'theta_max': hyperparameters.get('theta_max', 35), 
            'tc_theta': hyperparameters.get('tc_theta', 1e4), 
            'e_exc': hyperparameters.get('e_exc', 0), 
            'e_inh': hyperparameters.get('e_inh', -100)
        })

        self.INDEX_FIRED_OUTPUT = 0
        self.INDEX_V = 1
        self.INDEX_VTHR = 2
        self.INDEX_FIRED_V = 3
        self.INDEX_TLAST = 4
        self.INDEX_TCOUNT = 5
        self.INDEX_THETA = 5

    def initialize(self):
        self._states = [None, self.vreset * np.ones(self.n_neuron), self.init_vthr, None, 0, 0, None]
        
    def reset_states(self, random_state=False): 
        states = self.states
        if random_state: 
            states[self.INDEX_V] = self.vreset + np.random.rand(self.n_neuron) * (self.states[self.INDEX_VTHR] - self.vreset) 
        else: 
            states[self.INDEX_V] = self.vreset * np.ones(self.n_neuron)
            states[self.INDEX_VTHR] = self.init_vthr
            states[self.INDEX_THETA] = np.zeros(self.n_neuron) 
            states[self.INDEX_TLAST] = 0 
            states[self.INDEX_TCOUNT] = 0 
        self.cache_states(states)
    
    def pseudo_update_states(self, u):
        g_exc = u[0]
        g_inh = u[1]
        states = self.states
        theta = states[self.INDEX_THETA]
        tlast = states[self.INDEX_TLAST]
        tcount = states[self.INDEX_TCOUNT]
        vthr = states[self.INDEX_VTHR]
        v = states[self.INDEX_V]
        
        I_synExc = g_exc * (self.e_exc - v) 
        I_synInh = g_inh * (self.e_inh - v) 
        
        dv = (self.vrest - v + I_synExc + I_synInh) / self.tc_m 
        v = v + ((self.dt * tcount) > (tlast + self.tref)) * dv * self.dt 
        s = 1 * (v >= vthr) #発火時は1, その他は0の出力
        
        # 閾値の更新
        theta = (1 - self.dt / self.tc_theta) * theta + self.theta_plus * s 
        theta = np.clip(theta, 0, self.theta_max) 
        vthr = theta + self.init_vthr 
        tlast = tlast * (1 - s) + self.dt * tcount * s 
        v = v * (1 - s) + self.vpeak * s #閾値を超えると膜電位をvpeakにする
        v_ = v #発火時の電位も含めて記録するための変数
        v = v * (1 - s) + self.vreset * s #発火時に膜電位をリセット
        tcount += 1
        
        self.cache_states([s, v, vthr, v_, tlast, tcount, theta]) 
        return self.cached_states
        
    @property
    def dt(self):
        return self.hyperparameters['dt']
    
    @property
    def vreset(self):
        return self.hyperparameters['vreset']
    
    @property
    def vpeak(self):
        return self.hyperparameters['vpeak']
    
    @property
    def init_vthr(self):
        return self.hyperparameters['init_vthr']
    
    @property
    def theta_max(self):
        return self.hyperparameters['theta_max']
    
    @property
    def tc_theta(self):
        return self.hyperparameters['tc_theta']
    
    @property
    def e_exc(self):
        return self.hyperparameters['e_exc']
    
    @property
    def e_inh(self):
        return self.hyperparameters['e_inh']   
    
    @property
    def theta_plus(self):
        return self.hyperparameters['theta_plus']   
    
    @property
    def tref(self):
        return self.hyperparameters['tref']   
        
    @property
    def tc_m(self):
        return self.hyperparameters['tc_m']   
    
    @property
    def vrest(self):
        return self.hyperparameters['vrest']
    