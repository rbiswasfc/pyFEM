import os
import sys
import numpy as np
import pandas as pd
from operator import add, itemgetter

from node import Node

from data_loader import MeshDataLoader
from q8_element import FiniteElementQ8
from material_model import MaterialModel



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

        # initialize displacement field
        self.disp = np.zeros(shape = (self.n_dofs, 1))

        # initialize last converged displacement field
        self.disp0 = np.zeros(shape = (self.n_dofs, 1))

        self.left, self.right, self.top, self.bottom = self.get_domain_size()
    
    def get_domain_size(self):
        """
        find domain length scales
        """
        L, R, T, B = 0, 0, 0, 0
        
        for node in self.nodes:
            x, y = node.org_coords
            if x < L:
                L = x
            if x > R:
                R = x
            if y > T:
                T = y
            if y < B:
                B = y
        return L, R, T, B
    
    @property
    def disp_delta(self):
        return self.disp - self.disp0
        


    def set_disp_field(self, cur_disp):
        assert cur_disp.shape == (self.n_dofs, 1)
        self.disp = cur_disp
    
    def set_last_converged_disp_field(self):
        self.disp0 = self.disp 

    def update_nodes(self):
        """
        update displacement field of each node based on global displacement field
        """
        for node in self.nodes:
            dof_id = node.dof_id
            node.disp = self.disp[dof_id].flatten().tolist()

    def _find_nodes_along_axes(self, val = 0.0, direction = 'x', rel_tol = 1e-4):
        """
        function to find nodes in the axial directions
        @params:
        val -> float:
            nodes will be searched along x = val or y= val
        direction -> str:
            axial direction along which nodes will be searched
        rel_tol -> float:
            relative tolerance to match coordinates
        :return
        node_ids -> List:
            list of nodes ids that satisfies the condition set by 
            val and direction parameters
        """
        valid_dirs = ['x', 'y']
        assert direction in valid_dirs
        node_ids = []

        if direction == 'y': # search along x = val
            tol = rel_tol * (self.right - self.left)
            for node in self.nodes:
                if abs(node.org_coords[0] - val) < tol:
                    node_ids.append(int(node.node_id))
        else: # search along y = val
            tol = rel_tol * (self.top - self.bottom)
            for node in self.nodes:
                if abs(node.org_coords[1] - val) < tol:
                    node_ids.append(int(node.node_id))
        return node_ids

    def get_boundary_nodes(self, key):

        if key == 'left':
            node_ids = self._find_nodes_along_axes(self.left, 'y')
        elif key == 'right':
            node_ids = self._find_nodes_along_axes(self.right, 'y')
        elif key == 'top':
            node_ids = self._find_nodes_along_axes(self.top, 'x')
        elif key == 'bottom':
            node_ids = self._find_nodes_along_axes(self.bottom, 'x')
        else:
            print('Error: invalid key given')
            raise RuntimeError

        nodes = [self.nodes[idx] for idx in node_ids]
        return nodes

    def get_boundary_dofs(self, key, dof_dir = 'x'):
        """
        dof_dir -> str:
            if dof_dir is 1, dofs along x direction will be returned
        """
        valid_keys = ['left', 'right', 'top', 'bottom']
        assert key in valid_keys
        nodes = self.get_boundary_nodes(key)
        if dof_dir == 'x':
            dofs = [int(node.dof_id[0]) for node in nodes]
        elif dof_dir == 'y':
            dofs = [int(node.dof_id[1]) for node in nodes]
        else:
            print('invalid dof dir')
            raise RuntimeError
        return dofs

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

    domain.disp[0,0] = 1.0
    domain.disp[2,0] = 0.33
    domain.update_nodes()
