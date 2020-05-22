import os
import sys
import numpy as np 
import pandas as pd 

from node import Node
from data_loader import MeshDataLoader
from make_domain import Domain
from q8_element import FiniteElementQ8
from fea_solver import FESolver
from material_model import MaterialModel
from copy import deepcopy
from fea_solver import FESolver

if __name__ == '__main__':
    
    # load node and element data
    filepath = './datasets/mesh_data.xlsx'
    sheet = 'homogenized_localization'
    data_loader = MeshDataLoader(filepath, sheet)
    df_nodes = data_loader.get_node_data('C10', 'D157')
    df_elements = data_loader.get_element_data('B159', 'I197')
    material = MaterialModel()

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
        element = FiniteElementQ8(node_id, node_ids, df_nodes, material)
        elements.append(element)
    
    # construct domain
    domain = Domain(nodes, elements)

    # problem description
    num_steps = 4
    ubar = [0.0125, 0.0125, 0.025, 0.05] 
    assert len(ubar) == num_steps

    # boundary conditions
    bc = {"bc_fixed": [('left', 'x'), ('top', 'y')], 
          "bc_prescribed" : [('right', 'x')]}
    
    # stores solution results
    domain_frames = [domain]

    for step in range(num_steps):
        base_state = domain_frames[-1]
        fe_solver = FESolver(step, base_state, bc, ubar)
        final_state = fe_solver.execute()
        domain_frames.append(deepcopy(final_state))