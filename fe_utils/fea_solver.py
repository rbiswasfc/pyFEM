import os
import sys
import numpy as np
import pandas as pd 
from copy import deepcopy
from scipy.sparse import csr_matrix

from node import Node

from data_loader import MeshDataLoader
from q8_element import FiniteElementQ8
from material_model import MaterialModel
from make_domain import Domain

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

        self.strK = None # structural K matrix
        self.intF = None # internal nodal force vector

        self.netK = None
        self.netF = None

        self.fixed_dofs = None
        self.prescribed_dofs = None

    def configure_updated_state(self):
        
        # displacement field
        self.updated_state.set_last_converged_disp_field(self.base_state.disp)
        #self.updated_state.disp = self.base_state.disp

        # update last converged stress-strain field
        for i, element in enumerate(self.updated_state.elements):
            element.update_last_converged_node_disp(self.base_state.elements[i].disp_u)
            element.update_last_converged_stress(self.base_state.elements[i].stress_G)
            element.update_last_converged_strain(self.base_state.elements[i].strain_G)

    def initialize_field_variables(self):
        
        self.du = np.zeros(shape = (self.num_dof, 1))
        self.du0 = np.zeros(shape = (self.num_dof, 1))

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
    
    def solve_du(self):
        invK = np.linalg.inv(self.netK)
        du = np.matmul(invK, self.netF)
        return du
        
    
    def loop_over_elements(self):
        
        # perform a loop ever all elements
        if self.iteration == 0:
            self.du = self.get_initial_du_estimate()
        else:
            self.du = self.solve_du()
        # update domain 
        self.updated_state.update_disp_field(self.du)
        self.updated_state.update_nodes()
        

        for iel in range(self.num_elem):
            this_element = self.updated_state.elements[iel]
            du_elem = self.du[this_element.global_dof_ids].flatten()
            du_elem = np.expand_dims(du_elem, axis = -1)
            #print(du_elem.shape)
            K_elem, fint_elem, stress, strain = self.loop_over_gauss_points(this_element, du_elem)
            
            self.KC[iel] = K_elem
            self.fintC[iel] = fint_elem
            self.fluxC[iel] = np.sum(np.absolute(fint_elem[:,0]))

            this_element.update_node_disp(du_elem)
            this_element.update_stress(stress)
            this_element.update_strain(strain)
            
            self.updated_state.elements[iel] = this_element


    def loop_over_gauss_points(self, this_element, du_elem):
        strain_prev = this_element.strain_G
        tmp_strainG = np.zeros(shape = (4,this_element.ngp))

        for this_gp in range(this_element.ngp):
            J0 = this_element.jacobian_G[:,:,this_gp]
            BM = this_element.BM_G[:,:,this_gp]
            tmp_strainG[:,this_gp] = strain_prev[:,this_gp] + np.matmul(BM, du_elem).squeeze()
        
        K_elem, fint_elem, tmp_stressG = this_element.compute_response(tmp_strainG)
        return K_elem, fint_elem, tmp_stressG, tmp_strainG

    def get_global_stiffness_matrix(self):
        
        # stiffness matrix
        tmpI = []
        tmpJ = []
        tmpX = []

        for iel in range(self.num_elem):
            sctrKE = self.updated_state.elements[iel].global_dof_ids
            s = len(sctrKE)
            KE = self.KC[iel]
            #set_trace()
            for m in range(s):
                for n in range(s):
                    tmpI.append(sctrKE[m])
                    tmpJ.append(sctrKE[n])
                    tmpX.append(KE[m,n])

        tmpI, tmpJ, tmpX = np.array(tmpI), np.array(tmpJ), np.array(tmpX)

        K = csr_matrix((tmpX, (tmpI, tmpJ)), shape=(self.num_dof, self.num_dof)).toarray()
        return K
    
    def get_global_force_vector(self):
        # stiffness matrix
        tmpI = []
        tmpJ = []
        tmpX = []

        for iel in range(self.num_elem):
            sctrKE = self.updated_state.elements[iel].global_dof_ids
            s = len(sctrKE)
            fintE = self.fintC[iel]
            #set_trace()
            for m in range(s):
                tmpI.append(sctrKE[m])
                tmpJ.append(0)
                tmpX.append(fintE[m,0])

        tmpI, tmpJ, tmpX = np.array(tmpI), np.array(tmpJ), np.array(tmpX)

        fint = csr_matrix((tmpX, (tmpI, tmpJ)), shape=(self.num_dof, 1)).toarray()
        return fint

    def impose_bc(self):
        """
        imposes boundary conditions
        """
        bc_fixed = self.bc["bc_fixed"]
        bc_prescribed = self.bc["bc_prescribed"]

        fixed_dofs = []
        prescribed_dofs = []
        
        for boundary, axis in bc_fixed:
            dofs = self.base_state.get_boundary_dofs(boundary, axis)
            fixed_dofs.extend(dofs)

        for boundary, axis in bc_prescribed:
            dofs = self.base_state.get_boundary_dofs(boundary, axis)
            prescribed_dofs.extend(dofs)
        
        self.fixed_dofs = fixed_dofs
        self.prescribed_dofs = prescribed_dofs

        netK = self.strK.copy()
        netF = -self.intF.copy()

        bcwt = 1.0

        for dof in fixed_dofs:
            netK[dof,:] = 0
            netK[:,dof] = 0
            netK[dof, dof] = bcwt
        
        for dof in prescribed_dofs:
            netK[dof,:] = 0
            netK[:,dof] = 0
            netK[dof, dof] = bcwt 

        # External force vector due to specified boundary conditions        
        if ((self.step ==0) & (self.iteration == 0)):
            mm = np.zeros(shape = (self.num_dof,1))
            for dof in prescribed_dofs:
              mm[dof,0] = self.ubar[self.step]
            # force due to known displacements
            fm = -np.matmul(self.strK, mm)
            #print('shape of fm:')
            #print(fm.shape)        
            netF = netF + fm

        for dof in fixed_dofs:
            netF[dof,0] = 0

        if self.iteration == 0:
            if (self.step == 0):
              for dof in prescribed_dofs:
                    netF[dof] = self.ubar[self.step]
            else:
              for dof in prescribed_dofs:
                    netF[dof] = 0 # already added towards total displacement
        else:
            for dof in prescribed_dofs:
                netF[dof] = 0

        return netK, netF

    def check_convergence(self):
        
        """
        check for convergence
        """
        
        Rf = -self.intF.copy()


        for dof in self.fixed_dofs:
            Rf[dof] = 0

        for dof in self.prescribed_dofs:
            Rf[dof] = 0
        
        ave_flux = 0.0
        for iel in range(self.num_elem):
            ave_flux = ave_flux + self.fluxC[iel]
        ave_flux = ave_flux/(self.num_elem*16.0)
        
        absRf = np.absolute(Rf)
        max_Rf, max_Rf_idx = absRf.max(), absRf.argmax()
        
        # Relative error in flux
        self.err_flux = max_Rf/(1e-8+ave_flux) # add a small number to avoid 0/0

        # Compute maximum increment in displacement in the step
        du0 = self.updated_state.disp_delta
        max_du0 = np.absolute(du0).max()
        
        # current iteration
        max_du = np.absolute(self.du).max()
        # Relative error in displacement
        self.err_disp = max_du/max_du0
        
        if(self.iteration == 0) & (self.step == 0):
            self.err_disp = 1

        print('Average Flux in step {} iteration {} = {:.4f}'.format(self.step, 
                            self.iteration, ave_flux))
        print('Largest Residual Flux = {:.4f} at node {} dof {}'.format(max_Rf, 
                int(round(max_Rf_idx/2)), max_Rf_idx%2+1))
        print('Relative Error in Flux in step {} iteration {} = {:.4f}'.format(self.step, 
            self.iteration, self.err_flux))
        print('Relative Error in Disp in step {} iteration {} = {:.4f}'.format(self.step, 
        self.iteration, self.err_disp))

    def update_domain(self):
        """
        elements will be updated in-place during iterations
        """
        self.updated_state.update_disp_field(self.du)
        self.updated_state.update_nodes()
    
    def execute(self):

        while ((self.err_flux > self.tol_flux)| (self.err_disp > self.tol_disp)):
            if self.check_max_iter():
                print('Error: no convergence after {} iterations'.format(self.max_iter))
                raise RuntimeError
            self.loop_over_elements()
            self.strK = self.get_global_stiffness_matrix()
            self.intF = self.get_global_force_vector()
            self.netK, self.netF = self.impose_bc()
            self.check_convergence()
            self.iteration += 1



    
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

    # solver
    # boundary conditions
    bc = {"bc_fixed": [('left', 'x'), ('top', 'y')], 
          "bc_prescribed" : [('right', 'x')]}
    fe_solver = FESolver(0, domain, bc, [0.012, 0.012])
    fe_solver.execute()