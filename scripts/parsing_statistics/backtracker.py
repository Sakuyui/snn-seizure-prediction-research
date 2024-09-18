
from context import CKYBacktrackingContext
from typing import Callable
import numpy as np
class BackTrackingHelper():
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
    def to_global_grammar_id(self, global_symbol_id_begin, global_symbol_id_right_1, global_symbol_id_right_2):
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
                gid = self.to_global_grammar_id(associate_symbol_id, most_possible_nonterminate_id)
                pre_terminate_processing_strategy(context, cell, child_context, gid)
            elif self.is_start(associate_symbol_id):
                start_processing_strategy(context, cell)
                most_possible_nonterminate_id = np.argmax(grammar['start'])
                gid = self.to_global_grammar_id(associate_symbol_id, most_possible_nonterminate_id)
                child_context = self.__back_track(parse, grammar, i, j, CKYBacktrackingContext(), start_processing_strategy, pre_terminate_processing_strategy,
                                  terminate_processing_strategy, non_termination_processing_strategy, self.global_id_non_terminate(most_possible_nonterminate_id), keep_cnt)
                start_processing_strategy(context, cell, child_context, gid)
            else:
                raise ValueError
            return context_merging_strategy(context, child_context, None, gid)

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
                gid = self.to_global_grammar_id(associate_symbol_id, most_possible_nonterminate_ids[0], most_possible_nonterminate_ids[1])
                non_termination_processing_strategy(context, cell, context_children_1, context_children_2, gid)
                return context_merging_strategy(context, context_children_1, context_children_2, gid)
            elif self.is_start(associate_symbol_id):
                most_possible_nonterminate_id = np.argmax(grammar['start'])
                context_children = self.__back_track(parse, grammar, i, j, context, start_processing_strategy, pre_terminate_processing_strategy,
                        terminate_processing_strategy, non_termination_processing_strategy, 
                self.global_id_non_terminate(most_possible_nonterminate_id), keep_cnt)
                gid = self.to_global_grammar_id(associate_symbol_id, most_possible_nonterminate_id, None)
                start_processing_strategy(context, cell, context_children, gid)
                return context_merging_strategy(context, context_children, None, gid)
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
        
def to_parse_tree(grammar, parse, embed_grammar_id = False):
        def start_processing_strategy(context, cell, children1, grammar_id):
            pass
        def pre_terminate_processing_strategy(context, cell, children1, grammar_id):
            pass
        def non_termination_processing_strategy(context, cell, children1, children2, grammar_id):
            pass
        
        def terminate_processing_strategy(context, cell, symbol_id):
            context['tree'] = {
                'root': symbol_id,
                'suc': []
            }

        def result_retrieving_strategy(context):
            return context['tree']
        
        def context_merging_strategy(context, context_1, context_2, grammar_id):
            grammar_id = {} if embed_grammar_id else {'gid': grammar_id}
            result_context = CKYBacktrackingContext()
            result_context['tree'] = {
                    'root': context['root_symbol_id'],
                    'suc': [context_1['tree'], None if context_2 is None else context_2['tree']]
            } | grammar_id
            return result_context
        helper = BackTrackingHelper()
        return helper._back_track(parse=parse, grammar=grammar, len=parse.shape[0],
            start_processing_strategy=start_processing_strategy, 
            pre_terminate_processing_strategy=pre_terminate_processing_strategy,
            terminate_processing_strategy=terminate_processing_strategy,
            non_termination_processing_strategy=non_termination_processing_strategy,
            associate_symbol_id=helper.global_id_start(0), context_merging_strategy=context_merging_strategy,
            result_retrieving_strategy=result_retrieving_strategy,
            result_retrieving_strategy=None, keep_cnt=1)
        
def to_grammar_derivation_sequence(parse_tree):
    result_grammar_derivation_sequence = []
    def dfs(tree):
        if not tree:
            return
        if 'g_id' in tree:
            result_grammar_derivation_sequence.append(tree['g_id'])
        for suc in tree['suc']:
            dfs(suc)
    dfs(parse_tree)