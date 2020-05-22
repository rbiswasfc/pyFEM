import os
import sys
import numpy as np
import pandas as pd 

class FESolver(object):
    """
    Solves boundary value problems using FE. Updates the domain.
    """
    # class variables
    tol_flux = 5.0e-3
    tol_disp = 1.0e-2

    def __init__(self, domain, bc):
        """
        @params:
        domain -> object:
            object representing last converged state of the analysis domain
        bc -> :
            boundary conditions to be imposed
        """
        self.iteration = 0
        self.err_flux = 1.0
        self.err_disp = 1.0 

        self.domain = domain

        self.max_iter = 10

        # Macro stress
        stressC = [[] for _ in range (numelem)]
        # Macro strain
        strainC = [[] for _ in range (numelem)]
        # Macro strain of the last converged state
        strainCS = [[] for _ in range (numelem)]

    def loop_over_elements(self):
        
        num_elem = self.domain.n_elements
        # keep track of element stiffness matrices
        KC = [[] for _ in range (num_elem)]
        # keep track of internal force vectors
        fintC = [[] for _ in range (num_elem)]
        # keep track of error fluxes
        fluxC = [[] for _ in range (num_elem)]
        # perform a loop ever all elements
        for iel in range(num_elem):
            pass

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
    
    def execute_solver(self):

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
    pass