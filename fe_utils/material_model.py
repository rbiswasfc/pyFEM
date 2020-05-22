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
        C[0,0] = 1.111111e+09
        C[1,1] = 1.111111e+09
        C[0,1] = 2.777778e+08
        C[1,0] = 2.777778e+08
        C[2,2] = 4.166667e+08*0.5
        C[3,3] = 4.166667e+08*0.5
        C[2,3] = 4.166667e+08*0.5
        C[3,2] = 4.166667e+08*0.5
        
        return C

if __name__ == '__main__':
    pass 
