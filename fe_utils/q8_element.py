import os
import sys

import numpy as np 
import pandas as pd 

class FiniteElementQ8(object):
    '''
    class to implement Q8 finite element behavior
    '''
    def __init__(self, elem_id, node_ids, df_nodes):
        '''
        Function to implement element constructor interface

        @params:
        elem_id -> Int:
            unique identifier of the element
        nodes -> List:
            List of node ids maintaining the nodal connectivity sequence
        df_nodes -> pd.DataFrame:
            dataframe containing the node information for current alanlysis
        '''
        self.elem_type = 'Q8'
        self.analysis_type = 'PE' # plane strain
        self.elem_id = elem_id
        
        assert len(node_ids) == 8, "Error: Q8 element must have exactly 8 nodes"
        self.node_ids = node_ids
        self.node_coords = self._get_node_coords(df_nodes)
        

    def __repr__(self):
        return "2-D Q8 element under plane strain"
    
    def _get_node_coords(self, df_nodes):
        '''
        Function to get coordinates of the nodes for Q8 element

        @params:
        df_nodes -> pd.DataFrame:
            dataframe containing the node information for current alanlysis
        @retruns:
        node_coords -> np.ndarray:
            numpy array of shape (8,2) containing node coordinates
        '''
        node_coords = []
        for node in self.node_ids:
            X = df_nodes[df_nodes['node_id']==node].coord1.values[0]
            Y = df_nodes[df_nodes['node_id']==node].coord1.values[0]
            node_coords.append([X, Y])
        node_coords = np.array(node_coords)
        assert node_coords.shape == (8,2), "shape mismatch for node coordinates"
        return node_coords
    
    def initialize_sdvs(self):
        pass
    
    def get_interpolation_basis(self):
        pass

    def formulate_bmatrix(self):
        pass
    

if __name__ == '__main__':
    pass