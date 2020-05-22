import os
import sys

import numpy as np 
import pandas as pd 


class MaterialModel(object):
    def __init__(self, name = 'linear_elastic'):
        self.name = name 
    
    def compute_stress(self):
        pass
    
    def compute_tangent_stiffness(self):
        pass

if __name__ == '__main__':
    pass 
