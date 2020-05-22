import os
import sys
import numpy as np
import pandas as pd
from operator import add


class Domain(object):
    """
    class to provide abstraction for analysis domain 
    """

    def __init__(self, nodes, elements):
        """
        the domain is composed of elements and nodes
        @params:
        nodes -> List[objects]:
            a list containing all nodes of the analysis domain
        elements -> List[objects]:
            a list containing all elements of the analysis domain
        """
        self.elements = elements
        self.nodes = nodes

        self.n_nodes = len(nodes)
        self.n_elements = len(elements)

        self.n_dofs = 2*self.n_nodes
    
    def get_displacement_field(self):
        pass



if __name__ == '__main__':
    node = Node(0, [1.8, 2.7])
    node.disp = [0.2, 0.3]
    print(node.org_coords)
    print(node.cur_coords)
