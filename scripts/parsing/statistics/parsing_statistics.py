import numpy as np
from typing import Callable
from collections import deque, Counter
import scipy.stats
import scipy.stats._entropy
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

from ..context import CKYBacktrackingContext

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
    def single_tree_recursion_depth(self, parse_tree):
        symbol_last_appearance_record = {}
        symbol_recursion_depth_record = {}
        def dfs(tree, depth):
            if not tree:
                return
            symbol_id = tree['root']
            if symbol_id not in symbol_last_appearance_record:
                symbol_last_appearance_record[symbol_id] = deque()
            if symbol_id not in symbol_recursion_depth_record:
                symbol_recursion_depth_record[symbol_id] = []
            
            if len(symbol_last_appearance_record[symbol_id]) > 0:
                last_depth = symbol_last_appearance_record[symbol_id][-1]
                recursion_depth = depth - last_depth
                symbol_recursion_depth_record[symbol_id].append(recursion_depth)
            symbol_last_appearance_record[symbol_id].append(depth)
            
            for sub in tree['suc']:
                dfs(sub, depth + 1)
            symbol_last_appearance_record[symbol_id].pop()

        dfs(parse_tree, 0)
        return symbol_recursion_depth_record
    
    @classmethod
    def single_tree_symbol_entropy(self, parse_tree, symbol_count):
        counter_x = {}
        def dfs_counting(tree):
            if not tree:
                return
            self.count_increase(counter_x, tree['root'])
            for suc in tree['suc']:
                dfs_counting(suc)
        dfs_counting(parse_tree)
        total_count = sum(counter_x.values())
        distribution = \
            np.fromiter({key:counter_x[key] / total_count for key in counter_x}.values(), dtype=float)
        return scipy.stats._entropy.entropy(distribution)

    @classmethod
    def single_tree_avg_path_length(self, parse_tree):
        def dfs_calc_path_length(tree):
            result_lengths = []
            if not tree:
                return []
            if not tree['suc']:
                return [0]
            for suc in tree['suc']:
                lengths = 1 + np.array(dfs_calc_path_length(suc))
                result_lengths.extend(lengths)
            return result_lengths
        
        path_lengths = dfs_calc_path_length(parse_tree)
        if len(path_lengths) == 0:
            return 0
        
        return np.average(path_lengths)
    
    @classmethod
    def single_tree_layer_symbol_mutual_information(self, parse_tree, L=1):
        counter_x = {}
        counter_y = {}
        counter_xy = {}

        def bfs_collect_symbols(tree):
            if not tree:
                return []
            
            queue = deque([tree])
            for _ in range(L):
                queue_len = len(queue)
                for _ in range(queue_len):
                    current = queue.popleft()
                    for suc in current['suc']:
                        queue.append(suc)
            
            return [node['root'] for node in queue]
        
        def dfs(tree):
            if not tree:
                return
            self.count_increase(counter_x, tree['root'])
            L_succeed_node_root_symbol_id_list = bfs_collect_symbols(tree)
            for L_suc_root_symbol_id in L_succeed_node_root_symbol_id_list:
                self.count_increase(counter_y, L_suc_root_symbol_id)
                self.count_increase(counter_xy, f"{tree['root']}#{L_suc_root_symbol_id}")

            for suc in tree['suc']:
                dfs(suc)
            
        
        dfs(parse_tree)
        
        total_x = sum(counter_x.values())
        total_y = sum(counter_y.values())
        total_xy = sum(counter_xy.values())
        
        prob_x = {k: v / total_x for k, v in counter_x.items()}
        prob_y = {k: v / total_y for k, v in counter_y.items()}
        prob_xy = {k: v / total_xy for k, v in counter_xy.items()}
        
        mi = 0
        for x in prob_x:
            for y in prob_y:
                xy_key = f"{x}#{y}"
                if xy_key in prob_xy:
                    p_xy = prob_xy[xy_key]
                    p_x = prob_x[x]
                    p_y = prob_y[y]
                    if p_x > 0 and p_y > 0:
                        mi += p_xy * np.log(p_xy / (p_x * p_y))
                        
        return mi
    
    @classmethod
    def single_tree_layer_grammar_mutual_information(self, parse_tree, L=1):
        counter_x = {}
        counter_y = {}
        counter_xy = {}

        def bfs_collect_symbols(tree):
            if not tree:
                return []
            
            queue = deque([tree])
            for _ in range(L):
                queue_len = len(queue)
                for _ in range(queue_len):
                    current = queue.popleft()
                    for suc in current['suc']:
                        queue.append(suc)
            
            return [node['gid'] for node in queue]
        
        def dfs(tree):
            if not tree:
                return
            self.count_increase(counter_x, tree['root'])
            L_succeed_node_root_grammar_id_list = bfs_collect_symbols(tree)
            for L_suc_root_grammar_id in L_succeed_node_root_grammar_id_list:
                self.count_increase(counter_y, L_suc_root_grammar_id)
                self.count_increase(counter_xy, f"{tree['gid']}#{L_suc_root_grammar_id}")

            for suc in tree['suc']:
                dfs(suc)

        dfs(parse_tree)
        
        total_x = sum(counter_x.values())
        total_y = sum(counter_y.values())
        total_xy = sum(counter_xy.values())
        
        prob_x = {k: v / total_x for k, v in counter_x.items()}
        prob_y = {k: v / total_y for k, v in counter_y.items()}
        prob_xy = {k: v / total_xy for k, v in counter_xy.items()}
        
        mi = 0
        for x in prob_x:
            for y in prob_y:
                xy_key = f"{x}#{y}"
                if xy_key in prob_xy:
                    p_xy = prob_xy[xy_key]
                    p_x = prob_x[x]
                    p_y = prob_y[y]
                    if p_x > 0 and p_y > 0:
                        mi += p_xy * np.log(p_xy / (p_x * p_y))
        return mi
    
    @classmethod
    def single_tree_layer_count(self, parse_tree):
        cnt_layer = 0
        queue = deque([parse_tree])
        
        while len(queue) > 0:
            cnt_layer += 1
            layer_length = len(queue)
            for _ in range(layer_length):
                t = queue.popleft()
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
    
    @classmethod
    def single_parse_derivation_entropy(self, derivation_sequence):
        counter = Counter(derivation_sequence)
        total_count = len(derivation_sequence)
        probabilities = np.array([count / total_count for count in counter.values()])

        return scipy.stats._entropy.entropy(pk=probabilities)
        
    #! --------------------- single group multiple trees statistic characteristic ------------------ 
    @classmethod
    def multi_trees_mutual_information(self, parse_tree):
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