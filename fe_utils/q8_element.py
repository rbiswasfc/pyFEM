import os
import sys
sys.path.insert(0, '../')

import numpy as np 
import pandas as pd 
from copy import deepcopy
from element_utils import lagrange_basis, gauss_quadrature


class FiniteElementQ8(object):
    '''
    class to implement Q8 finite element behavior
    '''
    def __init__(self, elem_id, node_ids, df_nodes, gp_order = 2):
        '''
        Function to implement element constructor interface

        @params:
        elem_id -> Int:
            unique identifier of the element
        nodes -> List:
            List of node ids maintaining the nodal connectivity sequence
        df_nodes -> pd.DataFrame:
            dataframe containing the node information for current alanlysis
        gp_order -> Int:
            order of gauss quadrature
        '''
        self.elem_type = 'Q8'
        self.analysis_type = 'PE' # plane strain
        self.elem_id = elem_id
        
        assert len(node_ids) == 8, "Error: Q8 element must have exactly 8 nodes"
        self.node_ids = node_ids
        self.node_coords = self._get_node_coords(df_nodes)

        self.gp_order = gp_order
        self.ngp = gp_order**2 # total number of gauss points per element
        
        gp_data = self._get_gp_data()
        self.coord_G = gp_data[0]
        self.shape_func_G = gp_data[1]
        self.derivative_G = gp_data[2]
        self.jacobian_G = gp_data[3]
        self.BM_G = gp_data[4]

        # initialize field variables
        self._initialize_sdvs()
        self._initialize_node_disp()

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
            #print(self.node_ids)
            X = df_nodes[df_nodes['node_id']==node].coord1.values[0]
            Y = df_nodes[df_nodes['node_id']==node].coord2.values[0]
            #print("coord for {} are ({},{})".format(node, X, Y))
            node_coords.append([X, Y])
        node_coords = np.array(node_coords)
        assert node_coords.shape == (8,2), "shape mismatch for node coordinates"
        return node_coords

    def _get_gp_data(self):
        """
        function to compute coordinates,shape functions and its derivative 
        at gauss point

        @returns:
        coords_G -> np.ndarray:
            coordinates of the gauss points in physical coordinates
        shape_func_G -> np.ndarray:
            value of shape functions at gauss points
        derivative_G -> np.ndarray:
            derivative of shape functions wrt natural coordinates at the 
            gauss points
        """
        coords_G = np.zeros(shape = (2, self.ngp))
        shape_func_G = np.zeros(shape = (8, self.ngp))
        derivatives_G = np.zeros(shape = (8,2, self.ngp))
        jacobian_G = np.zeros(shape = (2,2, self.ngp))
        BM_G = np.zeros(shape = (4,16, self.ngp))

        points, weights = gauss_quadrature(self.gp_order)

        for this_gp in range(self.ngp):
            pt = points[this_gp,:]
            shape_func, derivative = lagrange_basis('Q8', pt)
            coord = np.matmul(shape_func.transpose(), self.node_coords)
            coords_G[:, this_gp] = coord
            shape_func_G[:, this_gp] = shape_func[:,0]
            derivatives_G[:,:,this_gp] = derivative
            jacobian = self.compute_jacobian(derivative)
            jacobian_G[:,:,this_gp] = jacobian
            BM = self.formulate_bmatrix(jacobian, derivative)
            BM_G[:,:,this_gp] = BM
        return coords_G, shape_func_G, derivatives_G, jacobian_G, BM_G

    
    def _initialize_sdvs(self):
        """
        initialize State Dependant Variables (SDVs) at the gauss points
        the SDVs are: stress tensor, strain tensor
        """
        self.stress_G = np.zeros(shape = (4,self.ngp))
        self.strain_G = np.zeros(shape = (4,self.ngp))

    def _initialize_node_disp(self):
        """
        initialize nodal displacements with zero
        """
        self.disp_u1 = np.zeros(shape = (8,1))
        self.disp_u2 = np.zeros(shape = (8,1))
    
    def compute_jacobian(self, dN):
        """
        function to compute jacobian at a point within the element
        @params:
        dN -> np.ndarray:
            derivative of shape functions at the point of interest

        @returns:
        J0 -> np.ndarray:
            jacobian at a point within the element 
        """
        J0 = np.matmul(self.node_coords.transpose(), dN)
        J0 = J0.transpose()
        return J0
    

    def formulate_bmatrix(self, J0, dN):
        """
        compute the starain displacement matrix (BMatrix) at a given point
        
        @params:
        J0 -> np.ndarray:   
            jacobian at the point of interest
        dN -> np.ndarray:
            derivative of shape functions at the point of interest
        
        @returns:
        BM -> np.ndarray:
            BMatrix at the point of interest
        """
        try:
            invJ0 = np.linalg.inv(J0)
            derivative_xy = (np.matmul(invJ0, dN.transpose())).transpose()

        except np.linalg.LinAlgError:
            print("Error: Jacobian not invertable")
            return

        BM = np.zeros(shape = (4, 16))
        for i in range(8):
            BM[0, 2*i] = derivative_xy[i,0]
            BM[1, 2*i+1] = derivative_xy[i,1]
            BM[2, 2*i] = derivative_xy[i,1]
            BM[3, 2*i+1] = derivative_xy[i,0]
        
        return BM
    

if __name__ == '__main__':
    
    #sys.path.insert(0, '../')
    from data_loader import MeshDataLoader

    filepath = './datasets/one_element_test.xlsx'
    sheet = 'Sheet1'
    data_loader = MeshDataLoader(filepath, sheet)
    df_nodes = data_loader.get_node_data('C10', 'D17')
    df_elements = data_loader.get_element_data('B19', 'I19')
    #print("node dataframe: ")
    #print(df_nodes)
    #print("element dataframe:" )
    #print(df_elements)


    tmp_el = df_elements.sample(1)
    elem_id = tmp_el.elem_id.values[0]
    print('element id = {}'.format(elem_id))
    node_ids = tmp_el.iloc[0, 1:].values.tolist()
    print("node ids for the element")
    print(node_ids)
    print('='*10)

    el = FiniteElementQ8(elem_id, node_ids,df_nodes)
    print(el)
    print(el.node_coords)

    dof_u = [0.0, 0.0, 
            0.1, 0.0,
            0.1, 0.0,
            0.0, 0.0,
            0.05, 0,
            0.1, 0.0,
            0.05, 0,
            0.0, 0.0
    ]
    dof_u = np.array(dof_u)
    dof_u = np.expand_dims(dof_u, axis = -1)
    strain = np.matmul(el.BM_G[:,:,0], dof_u)
    print("strain at GP 1: ")
    print(strain)
    el_2 = deepcopy(el) 
    x = (id(el) == id(el_2))
    print(x)



