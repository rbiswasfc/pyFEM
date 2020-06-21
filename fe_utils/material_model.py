import os
import sys

import numpy as np 
import pandas as pd 


class MaterialModel(object):
    def __init__(self, name = 'linear_elastic'):
        self.name = name 
    
    def compute_stress(self, strain):
        C = self.compute_tangent_stiffness()
        assert strain.shape == (4,1), "incompatible shape for strain"
        stress = np.matmul(C, strain)
        return stress
    
    def compute_tangent_stiffness(self, strain = None):
        
        C = np.zeros(shape = (4,4))
        C[0,0] = 1.863e+11
        C[0,1] = 7.076e+10
        C[0,2] = -9.181e+06
        C[0,3] = -1.055e+07

        C[1,0] = 7.076e+10
        C[1,1] = 1.864e+11
        C[1,2] = -4.488e+06
        C[1,3] = -4.088e+06
        
        C[2,0] = -9.181e+06
        C[2,1] = -4.488e+06
        C[2,2] = 5.180e+10
        C[2,3] = 5.180e+10

        C[3,0] = -1.055e+07
        C[3,1] = -4.088e+06
        C[3,2] = 5.179e+10
        C[3,3] = 5.179e+10
        
        return C

if __name__ == '__main__':
    pass 
