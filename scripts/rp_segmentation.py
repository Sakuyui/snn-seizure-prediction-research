from functools import cache
import numpy as np
class BaseSegmentGenerator(object):
    pass

class InfinitePhaseSpaceReonstructionBasedSegmentGenerator(BaseSegmentGenerator):
    def __init__(self, arg_lambda = 0.5, truncation = -1):
        self.arg_lambda = arg_lambda
        self.truncation = truncation
        
    def _observation_differs(self, D, t_i, t_j):
        return np.abs(D[t_i] - D[t_j])
    
    
    def _distance(self, D, t_i, t_j, dept = 0):
        _observation_differs = self._observation_differs(D, t_i, t_j)
        recursive_distance = 0 \
            if t_i == 0 or t_j == 0 or (self.truncation > 0 and dept >= self.truncation) \
            else self.arg_lambda * self._distance(t_i - 1, t_j - 1)
        distance = _observation_differs + recursive_distance
        return distance

    @cache
    def distance(self, D, t_i, t_j):
        return self._distance(D, t_i, t_j, 0)
    
    def calculate_recurrent_plot_points(self, D, epsilon = 0.01):
        length = D.shape[0]
        
        recurrent_plot_points = []
        for i in range(length):
            for j in range(length):
                if self.distance(D, i, j) < epsilon:
                    recurrent_plot_points.append((i, j))
                    
