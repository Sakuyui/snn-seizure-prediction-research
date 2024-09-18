import numpy as np
from typing import Callable
from collections import deque
'''
A parser is in the shape of (M, M, N), where M is the length of sequence and N 
is the count of non-termination symbols of the grammars.

A grammar is in the form of 
{
    'start': (1, N),
    'non_terminate': (N - P, N, N),
    'pre_terminate': (P, T)
}
'''

from context import CKYBacktrackingContext

class ParseTreeStatistics():
    def __init__(self):
        pass
    
    
    
    def count_increase(self, dictionary, key):
            if key not in dictionary:
                dictionary[key] = 0
            dictionary[key] += 1
            
    #! --------------------- single tree statistic characteristic ------------------ 
    @classmethod
    def single_tree_graph_entropy(self, parse_tree):
        raise NotImplemented
    
    @classmethod
    def single_tree_symbol_entropy(self, parse_tree):
        counter_x = {}
        def dfs_counting(tree):
            if not tree:
                return
            self.count_increase(counter_x, tree['root'])
            for suc in tree['suc']:
                dfs_counting(suc)
        dfs_counting(parse_tree)
        count_x = len(counter_x)
        distribution = \
            np.fromiter({key:counter_x[key] / count_x for key in counter_x}.values(), dtype=float)
        return np.multiply(distribution, np.log(distribution))

    @classmethod
    def single_tree_avg_path_length(self, parse_tree):
        def dfs_calc_path_length(tree):
            result_lengths = []
            if not tree:
                return [0]
            for suc in tree['suc']:
                lengths = 1 + dfs_calc_path_length(suc)
                result_lengths += lengths
            return result_lengths
        return np.average(dfs_calc_path_length(parse_tree))
    
    @classmethod
    def single_tree_layer_mutual_information(self, parse_tree, L=1):
        counter_x = {}
        counter_y = {}
        counter_xy = {}
        current_tree = parse_tree
        
            
        while not current_tree:
            self.count_increase(counter_x, current_tree['root'])
            
            L_succeed_node_root_symbol_id_list = []

            queue = deque()
            queue.append(current_tree)
            for _ in range(L):
                queue_len = len(queue)
                for _ in range(queue_len):
                    tree = queue.pop()
                    for suc in tree['suc']:
                        queue.append(suc)
            
            L_succeed_node_root_symbol_id_list = [tree['root'] for tree in list(queue)]
            for L_suc_root_symbol_id in L_succeed_node_root_symbol_id_list:
                self.count_increase(counter_y, L_suc_root_symbol_id)
                self.count_increase(counter_xy, f"{current_tree['root']}#{L_suc_root_symbol_id}")
        
        count_x = len(counter_x)
        counter_x = {k:counter_x[k] / count_x for k in counter_x}
        
        count_y = len(counter_y)
        counter_y = {k:counter_y[k] / count_y for k in counter_y}     
        
        count_xy = len(counter_xy)
        counter_xy = {k:counter_xy[k] / count_xy for k in counter_xy}
        
        mi = 0
        for x in counter_x:
            for y in counter_y:
                p_xy = counter_xy[f'{x}#{y}']
                mi += p_xy * np.log(p_xy / (x * y))
        return mi
    
    @classmethod
    def single_tree_layer_count(self, parse_tree):
        cnt_layer = 0
        queue = deque()
        queue.append(parse_tree)
        while len(queue) > 0:
            cnt_layer += 1
            layer_length = len(queue)
            for _ in range(layer_length):
                t = queue.pop()
                for suc in t['suc']:
                    queue.append(suc)
        return cnt_layer        
    
    @classmethod
    def single_tree_symbol_count(self, parse_tree):
        counter_x = {}
        def dfs_counting(tree):
            if not tree:
                return
            self.count_increase(counter_x, tree['root'])
            for suc in tree['suc']:
                dfs_counting(suc)
        dfs_counting(parse_tree)
        return counter_x
    
    
        
    #! --------------------- single group multiple trees statistic characteristic ------------------ 
    @classmethod
    def multi_trees_mutual_information(self, parse):
        raise NotImplemented
    
    
    #! --------------------- multiple group multiple trees statistic characteristic ------------------ 
    @classmethod
    def multi_trees_diff_group_significant_difference_test(self, parse_set_1, parse_set_2):
        raise NotImplemented
    
    @classmethod
    def multi_trees_diff_group_mutual_information_test(self, parse_set_1, parse_set_2):
        raise NotImplemented
    
    #! ----------------------------------- prediction evaluation -------------------------------------
    def mutual_information_analysis(self, parse_set, output):
        raise NotImplemented