from .context import CKYBacktrackingContext
from typing import Callable
import numpy as np
import sys
sys.path.append("../..")
sys.path.append("../")

from eeg_lang.scripts.parsers.comm_cky_parser.pcfgs.cnf_pcfg import MatrixAndBitmapBasedExplicitPreterminateCNFPCFG

class BackTrackingHelper():
    def __init__(self, grammar: MatrixAndBitmapBasedExplicitPreterminateCNFPCFG, 
                 terminate_processing_strategy: Callable,
                 pre_terminate_processing_strategy: Callable,
                 non_termination_processing_strategy: Callable,
                 start_processing_strategy: Callable,
                 context_merging_strategy: Callable,
                 result_retrieving_strategy: Callable) -> None:
        self.grammar: MatrixAndBitmapBasedExplicitPreterminateCNFPCFG = grammar
        self.terminate_processing_strategy = terminate_processing_strategy
        self.pre_terminate_processing_strategy = pre_terminate_processing_strategy
        self.start_processing_strategy = start_processing_strategy
        self.context_merging_strategy = context_merging_strategy
        self.non_termination_processing_strategy = non_termination_processing_strategy
        self.result_retrieving_strategy = result_retrieving_strategy
        
    def _N_id_to_global(self, N_id):
        """Maps non-terminal ID to global ID."""
        assert N_id < self.grammar.N
        return self.grammar.N2_global_sym_id(N_id) if N_id <self. grammar.N2 else self.grammar.P_global_sym_id(N_id)

    def _process_terminate(self, cell, associate_symbol_id):
        """Processes terminal symbols during backtracking."""
        child_context = CKYBacktrackingContext()
        self.terminate_processing_strategy(child_context, cell, associate_symbol_id)
        return child_context
    
    def _process_preterminate(self, parse, i, j, cell, context, associate_symbol_id):
        """Processes pre-terminal symbols during backtracking."""
        P_local_id = self.grammar.decode_to_local_P(associate_symbol_id)
        right_derivation_id1_local = np.argmax(self.grammar['pre_terminate'][P_local_id])
        right_derivation_id1_global = self.grammar.T_global_sym_id(right_derivation_id1_local)

        child_context = self.__back_track(parse=parse, i=i, j=j, 
            context=CKYBacktrackingContext(),
            associate_symbol_id=right_derivation_id1_global)
        gid = self.grammar.to_grammar_global_id(associate_symbol_id, right_derivation_id1_global)
        self.pre_terminate_processing_strategy(context, cell, child_context, gid)

        return gid, child_context
    
    def _process_start(self, parse, i, j, cell, context, associate_symbol_id):
        """Processes start symbols during backtracking."""
        right_derivation_id1_local = np.argmax(self.grammar['start'])
        right_derivation_id1_global = self._N_id_to_global(right_derivation_id1_local)

        child_context = self.__back_track(parse,self.grammar, i, j, CKYBacktrackingContext(),
            self.start_processing_strategy, self.pre_terminate_processing_strategy,
            self.terminate_processing_strategy, None,
            right_derivation_id1_global)
        gid = self.grammar.to_grammar_global_id(associate_symbol_id, right_derivation_id1_global)
        self.start_processing_strategy(context, cell, child_context, gid)

        return gid, child_context
    
    def _process_order_2non_termination(self, parse, i, j, context, cell, associate_symbol_id):
        """Processes non-terminal symbols and backtracks."""
        local_id = self.grammar.decode_global_sym_id(associate_symbol_id)
        rules = self.grammar['non_terminate'][local_id]
        right_derivation_ids_local = np.unravel_index(rules.argmax(), rules.shape)

        best_k, best_p = self._find_best_split(parse=parse, i=i, j=j, 
            right_derivation_ids_local=right_derivation_ids_local, 
            rules=rules)
        
        right_derivation_id1_global = self._N_id_to_global(right_derivation_ids_local[0])
        right_derivation_id2_global = self._N_id_to_global(right_derivation_ids_local[1])

        context_children_1 = self.__back_track(parse=parse, i=i, j=best_k, 
            context=CKYBacktrackingContext(),
            associate_symbol_id=right_derivation_id1_global)

        context_children_2 = self.__back_track(parse=parse, i=best_k + 1, j=j, 
            context=CKYBacktrackingContext(),
            associate_symbol_id=right_derivation_id2_global)

        gid = self.grammar.to_grammar_global_id(associate_symbol_id, 
                                                right_derivation_id1_global, 
                                                right_derivation_id2_global)
        self.non_termination_processing_strategy(context, cell, context_children_1, context_children_2, gid)

        return gid, context_children_1, context_children_2
    
    def _find_best_split(self, parse, i, j, right_derivation_ids_local, rules):
        """Finds the best split point (k) during backtracking for non-terminal symbols."""
        best_k, best_p = i, -np.inf
        for k in range(i + 1, j):
            p_grammar = rules[right_derivation_ids_local[0], right_derivation_ids_local[1]]
            p_left = parse[i, k, right_derivation_ids_local[0]]
            p_right = parse[i, k, right_derivation_ids_local[1]]
            p = p_grammar * p_left * p_right
            if p > best_p:
                best_k, best_p = k, p
        return best_k, best_p

    def __back_track(self, parse, i, j, context, associate_symbol_id, keep_cnt=1):
        """Recursively backtracks through the parse chart."""
        if j < i or j < 0 or i < 0:
            return None
        if keep_cnt > 1:
            raise NotImplementedError

        cell = parse[i, j]  # shape = (N)

        if i == j:  # Terminal and pre-terminal symbols
            if self.grammar.is_T(associate_symbol_id):
                return self._process_terminate(cell=cell, associate_symbol_id=associate_symbol_id)

            elif self.grammar.is_P(associate_symbol_id):
                gid, child_context = self._process_preterminate(parse=parse, i=i, j=j, cell=cell, context=context, associate_symbol_id=associate_symbol_id)

            elif self.grammar.is_S(associate_symbol_id):
                gid, child_context = self._process_start(parse=parse, i=i, j=j, cell=cell, context=context, associate_symbol_id=associate_symbol_id)

            else:
                print("Warning: encounter unexpected symbol in backtracking. This may cause incomplete back tracking.")
                return None

            return self.context_merging_strategy(context, child_context, None, gid)

        if i < j:  # Non-terminal symbols
            if self.grammar.is_N2(associate_symbol_id):
                gid, context_children_1, context_children_2 = self._process_order_2non_termination(parse=parse, i=i, j=j, context=context, cell=cell, associate_symbol_id=associate_symbol_id)
                return self.context_merging_strategy(context, context_children_1, context_children_2, gid)

            elif self.grammar.is_S(associate_symbol_id):
                right_derivation_id_local = np.argmax(self.grammar['start'])
                right_derivation_id_global = self._N_id_to_global(right_derivation_id_local)
                context_children = self.__back_track(parse=parse, i=i, j=j, context=context, associate_symbol_id=right_derivation_id_global)
                gid = self.grammar.to_grammar_global_id(associate_symbol_id, right_derivation_id_global)
                self.start_processing_strategy(context, cell, context_children, gid)
                return self.context_merging_strategy(context, context_children, None, gid)
            else:
                print("Warning: encounter unexpected symbol in backtracking. This may cause incomplete back tracking.")
    
    def _back_track(self, parse, length, context, associate_symbol_id, keep_cnt=1):
        """Main backtracking entry point."""
        context = CKYBacktrackingContext()
        result_context = self.__back_track(parse=parse, i=0, j=length-1, 
            context=context,
            associate_symbol_id=associate_symbol_id,
            keep_cnt=keep_cnt)
        return self.result_retrieving_strategy(result_context)
        
def to_parse_tree(grammar: MatrixAndBitmapBasedExplicitPreterminateCNFPCFG, parse, embed_grammar_id = False):
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
            if embed_grammar_id:
                context['tree'] |= {'gid': -1}

        def result_retrieving_strategy(context):
            return context['tree']
        
        def context_merging_strategy(context, context_1, context_2, grammar_id):
            wrapped_grammar_id = {} if not embed_grammar_id else {'gid': grammar_id}
            result_context = CKYBacktrackingContext()

            result_context['tree'] = {
                    'root': grammar.left_symbol_global_id(grammar_id),
                    'suc':  [ctx['tree'] for ctx in [context_1, context_2] if ctx is not None]
            } | wrapped_grammar_id
            return result_context
        
        helper = BackTrackingHelper(grammar=grammar, terminate_processing_strategy=terminate_processing_strategy,
                                    pre_terminate_processing_strategy=pre_terminate_processing_strategy,
                                    non_termination_processing_strategy=non_termination_processing_strategy,
                                    start_processing_strategy=start_processing_strategy,
                                    context_merging_strategy=context_merging_strategy,
                                    result_retrieving_strategy=result_retrieving_strategy)
        
        return helper._back_track(parse=parse, length=parse.shape[0], context=CKYBacktrackingContext(),
            associate_symbol_id=grammar.S_global_sym_id(), 
            keep_cnt=1)
        
def to_grammar_derivation_sequence(parse_tree):
    result_grammar_derivation_sequence = []
    def dfs(tree):
            if not tree:
                return
            if 'gid' in tree:
                result_grammar_derivation_sequence.append(tree['gid'])
            for suc in tree['suc']:
                dfs(suc)
    dfs(parse_tree)
    return result_grammar_derivation_sequence