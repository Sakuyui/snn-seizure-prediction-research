import numpy as np
from typing import Callable
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
                terminate_processing_strategy(context, cell, child_context, None)

            elif self.is_preterminate(associate_symbol_id):
                local_id = self.local_id_pre_terminate(associate_symbol_id)
                most_possible_terminate_id = np.argmax(grammar['non_terminate'][local_id])
                child_context = self.__back_track(parse, grammar, i, j, CKYBacktrackingContext(), start_processing_strategy, pre_terminate_processing_strategy,
                                  terminate_processing_strategy, non_termination_processing_strategy, self.global_id_terminate(most_possible_terminate_id), keep_cnt)
                pre_terminate_processing_strategy(context, cell, child_context, None)
            elif self.is_start(associate_symbol_id):
                start_processing_strategy(context, cell)
                most_possible_nonterminate_id = np.argmax(grammar['start'])
                child_context = self.__back_track(parse, grammar, i, j, CKYBacktrackingContext(), start_processing_strategy, pre_terminate_processing_strategy,
                                  terminate_processing_strategy, non_termination_processing_strategy, self.global_id_non_terminate(most_possible_nonterminate_id), keep_cnt)
                start_processing_strategy(context, cell, child_context, None)
            else:
                raise ValueError
            return context_merging_strategy(context, child_context, None)

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

    def _back_track(self, parse, grammar, len, terminate_processing_strategy: Callable, 
                    non_termination_processing_strategy: Callable, result_retrieving_strategy: Callable, associate_symbol_id):
        context = CKYBacktrackingContext()
        self.__back_track(parse, grammar, 0, len - 1, context, terminate_processing_strategy, 
                          non_termination_processing_strategy, associate_symbol_id, True)
        return result_retrieving_strategy(context)
        
    def __init__(self):
        pass
    
    #! --------------------- single tree statistic characteristic ------------------ 
    @classmethod
    def single_tree_graph_entropy(self, parse):
        raise NotImplemented
    
    @classmethod
    def single_tree_avg_path_length(self, parse):
        raise NotImplemented
    
    @classmethod
    def single_tree_layer_mutual_information(self, parse):
        raise NotImplemented
    
    @classmethod
    def single_tree_layer_count(self, parse):
        raise NotImplemented
    
    @classmethod
    def single_tree_symbol_count(self, parse):
        raise NotImplemented
    
    
        
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