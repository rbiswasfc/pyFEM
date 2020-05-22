import os
import sys
import numpy as np 
import pandas as pd 

from data_loader import MeshDataLoader
from make_domain import Node, Domain
from q8_element import FiniteElementQ8
from fea_solver import FESolver

if __name__ == '__main__':
    
    # load node and element data
    filepath = './datasets/mesh_data.xlsx'
    sheet = 'homogenized_localization'
    data_loader = MeshDataLoader(filepath, sheet)
    df_nodes = data_loader.get_node_data('C10', 'D157')
    df_elements = data_loader.get_element_data('B159', 'I197')

    # construct nodes
    nodes = []
    for index, row in df_nodes.iterrows():
        node_id, coord_1, coord_2 = row
        node = Node(node_id, [coord_1, coord_2])
        nodes.append(node)

    # construct elements
    elements = []
    for index, row in df_elements.iterrows():
        elem_id, *node_ids = row
        element = FiniteElementQ8(node_id, node_ids, df_nodes)
        elements.append(element)
    
    # construct domain
    domain = Domain(nodes, elements)

    # perform analysis


