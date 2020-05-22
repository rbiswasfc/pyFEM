import os
import sys
import numpy as np
import pandas as pd 
from copy import deepcopy

from node import Node

from data_loader import MeshDataLoader
from q8_element import FiniteElementQ8
from material_model import MaterialModel

class FESolver(object):
    """
    Solves boundary value problems using FE. Updates the domain.
    """
    # class variables
    tol_flux = 5.0e-3
    tol_disp = 1.0e-2

    def __init__(self, step, base_state, bc, ubar, max_iter = 10):
        """
        @params:
        domain -> object:
            object representing last converged state of the analysis domain
        bc -> :
            boundary conditions to be imposed
        """
        self.step = step
        
        self.base_state = base_state
        self.updated_state = deepcopy(base_state)
        self.configure_updated_state()

        self.bc = bc
        self.ubar = ubar

        self.iteration = 0
        self.err_flux = 1.0
        self.err_disp = 1.0 
        self.max_iter = max_iter

        self.num_elem = self.base_state.n_elements
        self.num_nodes = self.base_state.n_nodes
        self.num_dof = self.base_state.n_dofs

        self.initialize_field_variables()

    def configure_updated_state(self):
        
        # displacement field
        self.updated_state.disp0 = self.base_state.disp
        self.updated_state.disp = self.base_state.disp

        # update last converged stress-strain field
        for i, element in enumerate(self.updated_state.elements):
            element.update_last_converged_stress(self.base_state.elements[i].stress_G)
            element.update_last_converged_strain(self.base_state.elements[i].strain_G)

    def initialize_field_variables(self):
        
        self.du = np.zeros(shape = (self.num_dof, 1))
        self.du0 = np.zeros(shape = (self.num_dof, 1))
        
        #self.KC = [[] for _ in range (self.num_elem)]
        #self.fintC = [[] for _ in range (self.num_elem)]
        #self.fluxC = [[] for _ in range (self.num_elem)]

        # keep track of element stiffness matrices
        self.KC = [[] for _ in range (self.num_elem)]
        # keep track of internal force vectors
        self.fintC = [[] for _ in range (self.num_elem)]
        # keep track of error fluxes
        self.fluxC = [[] for _ in range (self.num_elem)]
    
    def get_initial_du_estimate(self):
        """
        get initial estimate for displacement increment in the current step
        """
        if self.step > 0:
            ratio = (self.ubar[self.step])/(self.ubar[self.step-1])
        else:
            ratio = 1.0

        prev_du = self.base_state.disp_delta
        du_star = ratio*prev_du
        return du_star
        
    
    def loop_over_elements(self):
        
        # perform a loop ever all elements
        self.du = self.get_initial_du_estimate()
        
        for iel in range(self.num_elem):
            du_elem = self.du[self.updated_state.elements[iel].global_dof_ids].flatten()

    def loop_over_gauss_points(self):
        pass

    def get_stiffness_matrix(self):
        pass
    def get_internal_force_vector(self):
        pass
    def check_convergence(self):
        pass
    def update_domain(self):
        pass
    def perform_one_iteration(self):
        pass
    
    def execute(self):

        while ((self.err_flux > tol_flux)| (self.err_disp > tol_disp)):
            if self.check_max_iter():
                print('Error: no convergence after {} iterations'.format(self.max_iter))
                raise RuntimeError
            self.loop_over_elements()

    
    def check_max_iter(self):
        """
        function to check if maximum number of iteration steps is reached
        """
        if self.iteration >= self.max_iter:
            return True
        else:
            return False

        

if __name__ == "__main__":
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