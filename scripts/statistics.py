import numpy as np

class ParseTreeStatistics():
    def __init__(self):
        pass
    
    #! --------------------- single tree statistic characteristic ------------------ 
    @classmethod
    def single_tree_graph_entropy(self, parse):
        raise NotImplemented
    
    @classmethod
    def single_tree_avg_path_length(self, parse):
        raise NotImplemented
    
    @classmethold
    def single_tree_layer_mutual_information(self, parse):
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
    
    