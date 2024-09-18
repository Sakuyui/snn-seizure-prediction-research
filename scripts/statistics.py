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

class CKYBacktrackingContext():
    def __init__(self):
        self._context = {}
        
    def __getitem__(self, index):
        return self._context[index]
    def __setitem__(self, key, value):
        self._context[key] = value

class ParseTreeStatistics():
    def is_terminate(self, symbol_id):
        raise NotImplementedError
    def is_preterminate(self, symbol_id):
        raise NotImplementedError
    def is_start(self, symbol_id):
        raise NotImplementedError
    def is_non_terminate(self, symbol_id):
        raise NotImplementedError
    
    def global_id_non_terminate(self, non_terminate_symbol_id):
        raise NotImplementedError
    def global_id_terminate(self, terminate_symbol_id):
        raise NotImplementedError
    def global_id_start(self, start_symbol_id):
        raise NotImplementedError
    def local_id_pre_terminate(self, pre_terminate):
        raise NotImplementedError
    def local_id_non_terminate(self, non_terminate):
        raise NotImplementedError
    
    def __back_track(self, parse, grammar, i, j, context, 
                     start_processing_strategy: Callable,
                     pre_terminate_processing_strategy: Callable, 
                     terminate_processing_strategy: Callable, 
                    non_termination_processing_strategy: Callable, associate_symbol_id,  
                    context_merging_strategy: Callable,
                    keep_cnt=1):
        if j < i or j < 0 or i < 0:
            return []
        if keep_cnt > 1:
            raise NotImplementedError
        
        cell = parse[i, j] # shape = (N)

        if i == j:
            if self.is_terminate(associate_symbol_id) or self.is_non_terminate(associate_symbol_id):
                child_context = CKYBacktrackingContext()
                terminate_processing_strategy(child_context, cell, associate_symbol_id)

            elif self.is_preterminate(associate_symbol_id):
                local_id = self.local_id_pre_terminate(associate_symbol_id)
                most_possible_terminate_id = np.argmax(grammar['non_terminate'][local_id])
                child_context = self.__back_track(parse, grammar, i, j, CKYBacktrackingContext(), start_processing_strategy, pre_terminate_processing_strategy,
                                  terminate_processing_strategy, non_termination_processing_strategy, self.global_id_terminate(most_possible_terminate_id), keep_cnt)
                pre_terminate_processing_strategy(context, cell, child_context)
            elif self.is_start(associate_symbol_id):
                start_processing_strategy(context, cell)
                most_possible_nonterminate_id = np.argmax(grammar['start'])
                child_context = self.__back_track(parse, grammar, i, j, CKYBacktrackingContext(), start_processing_strategy, pre_terminate_processing_strategy,
                                  terminate_processing_strategy, non_termination_processing_strategy, self.global_id_non_terminate(most_possible_nonterminate_id), keep_cnt)
                start_processing_strategy(context, cell, child_context)
            else:
                raise ValueError
            return context_merging_strategy(context, child_context)

        if i < j:
            if self.is_terminate(associate_symbol_id) or self.is_preterminate(associate_symbol_id):
                return []
            elif self.is_non_terminate(associate_symbol_id):
                non_termination_processing_strategy(context, cell)
                local_id = self.local_id_non_terminate(associate_symbol_id)
                rules = grammar['non_terminate'][local_id]
                most_possible_nonterminate_ids = np.unravel_index(rules.argmax(), rules.shape)
                best_k = i
                best_p = -np.inf
                for k in range(i + 1, j):
                    p_grammar = rules[most_possible_nonterminate_ids[0], most_possible_nonterminate_ids[1]]
                    p_left = parse[i, k, most_possible_nonterminate_ids[0]]
                    p_right = parse[i, k, most_possible_nonterminate_ids[1]]
                    p = p_grammar * p_left * p_right
                    if p > best_p:
                        best_k = k
                        best_p = p
                
                context_children_1 = self.__back_track(parse, grammar, i, best_k, CKYBacktrackingContext(), 
                        start_processing_strategy, pre_terminate_processing_strategy,
                        terminate_processing_strategy, non_termination_processing_strategy, 
                    self.global_id_non_terminate(most_possible_nonterminate_ids[0]), keep_cnt)
                    
                context_children_2 = self.__back_track(parse, grammar, best_k + 1, j, CKYBacktrackingContext(),
                        start_processing_strategy, pre_terminate_processing_strategy,
                        terminate_processing_strategy, non_termination_processing_strategy, 
                    self.global_id_non_terminate(most_possible_nonterminate_ids[1]), keep_cnt)
                non_termination_processing_strategy(context, cell, context_children_1, context_children_2)
                return context_merging_strategy(context, context_children_1, context_children_2)
            elif self.is_start(associate_symbol_id):
                most_possible_nonterminate_id = np.argmax(grammar['start'])
                context_children = self.__back_track(parse, grammar, i, j, context, start_processing_strategy, pre_terminate_processing_strategy,
                        terminate_processing_strategy, non_termination_processing_strategy, 
                self.global_id_non_terminate(most_possible_nonterminate_id), keep_cnt)
                start_processing_strategy(context, cell, context_children)
                return context_merging_strategy(context, context_children)
            else:
                raise ValueError

    def _back_track(self, parse, grammar, len, context, 
                     start_processing_strategy: Callable,
                     pre_terminate_processing_strategy: Callable, 
                     terminate_processing_strategy: Callable, 
                    non_termination_processing_strategy: Callable, associate_symbol_id,  
                    context_merging_strategy: Callable,
                    result_retrieving_strategy: Callable,
                    keep_cnt=1):
        context = CKYBacktrackingContext()
        self.__back_track(parse=parse, grammar=grammar, i=0, j=len-1,
                          start_processing_strategy=start_processing_strategy,
                          pre_terminate_processing_strategy=pre_terminate_processing_strategy,
                          terminate_processing_strategy=terminate_processing_strategy,
                          non_termination_processing_strategy=non_termination_processing_strategy,
                          associate_symbol_id=associate_symbol_id,
                          context_merging_strategy=context_merging_strategy,
                          keep_cnt=keep_cnt)
        return result_retrieving_strategy(context)
        
    def __init__(self):
        pass
    
    def to_parse_tree(self, grammar, parse):
        def start_processing_strategy(context, cell, children1):
            pass
        def pre_terminate_processing_strategy(context, cell, children1):
            pass
        def non_termination_processing_strategy(context, cell, children1, children2):
            pass
        
        def terminate_processing_strategy(context, cell, symbol_id):
            context['tree'] = {
                'root': symbol_id,
                'suc': []
            }
        
        def result_retrieving_strategy(context):
            return context['tree']
        
        def context_merging_strategy(context, context_1, context_2):
            result_context = CKYBacktrackingContext()
            result_context['tree'] = {
                    'root': context['root_symbol_id'],
                    'suc': [context_1['tree'], None if context_2 is None else context_2['tree']]
            }
            return result_context

        return self._back_track(parse=parse, grammar=grammar, len=parse.shape[0],
            start_processing_strategy=start_processing_strategy, 
            pre_terminate_processing_strategy=pre_terminate_processing_strategy,
            terminate_processing_strategy=terminate_processing_strategy,
            non_termination_processing_strategy=non_termination_processing_strategy,
            associate_symbol_id=self.global_id_start(0), context_merging_strategy=context_merging_strategy,
            result_retrieving_strategy=result_retrieving_strategy,
            result_retrieving_strategy=None, keep_cnt=1)
    
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