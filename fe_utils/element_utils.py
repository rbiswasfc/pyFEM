import os
import sys

import numpy as np
import pandas as pd

def lagrange_basis(elem_type, query_point, dim = 2):
    """
    Function to compute shape functions and its derivatives
    (lagrange interpolation)

    @params:
    elem_type -> str:
        type of element to be used for meshing
    query_point -> List:
        natural coordinates
    dim -> Int:
        dimension of computational domain

    @returns:
    shape_fns -> np.ndarray:
        shape functions of proper element type
        for Q8 element the array shape is (8,1)
    derivatives -> np.ndarray:
        derivative of shape functions wrt natural coorsinates
        for Q8 element it has shape of (8,2)
    """
    assert len(query_point) == dim, "dimension mismatch b/w coord and dim"
    if dim != 2:
        raise NotImplementedError
  
    if elem_type == 'Q8':
        xi, eta = query_point[0], query_point[1]
        # shape functions
        shape_fns = [(1-xi)*(1-eta)*(1+xi+eta),
                    (1+xi)*(1-eta)*(1-xi+eta),
                    (1+xi)*(1+eta)*(1-xi-eta),
                    (1-xi)*(1+eta)*(1+xi-eta),
                    -2*(1-xi)*(1+xi)*(1-eta),
                    -2*(1+xi)*(1-eta)*(1+eta),
                    -2*(1-xi)*(1+xi)*(1+eta),
                    -2*(1-xi)*(1-eta)*(1+eta)
                    ]
        shape_fns = np.array(shape_fns)
        shape_fns = -0.25*shape_fns
        shape_fns = np.expand_dims(shape_fns, axis=-1)

        # spatial derivative of shape function
        derivatives = [
                        [(2*xi+eta)*(eta-1), (xi+2*eta)*(xi-1)],
                        [(2*xi-eta)*(eta-1), (xi-2*eta)*(xi+1)],
                        [-(2*xi+eta)*(eta+1), -(xi+2*eta)*(xi+1)],
                        [(-2*xi+eta)*(eta+1), (xi-2*eta)*-(xi-1)],
                        [2*(2*xi)*(1-eta), 2*(1-xi)*(1+xi)],
                        [-2*(1-eta)*(1+eta), 2*(1+xi)*(2*eta)],
                        [2*(2*xi)*(1+eta), -2*(1-xi)*(1+xi)],
                        [2*(1-eta)*(1+eta), 2*(1-xi)*(2*eta)]
                        ]
    
        derivatives = np.array(derivatives)
        derivatives = -0.25*derivatives
        
        assert shape_fns.shape == (8,1), "Error: shape func shape should be (8,1)"
        assert derivatives.shape == (8,2), "Error: derivative size should be (8,2)"

    if elem_type!= 'Q8':
        raise NotImplementedError

    return shape_fns, derivatives
##
def gauss_quadrature(order, dim =2):
    """
    function to provide integration points and associated weights for
    performing gauss quadrature

    @params:
    order -> Int:
        order of gauss quadrature
    dim -> Int:
        dimension of analysis domain
    
    @returns:
    points -> np.ndarray:
        an array with gauss point (natural) coordinates 
    weights -> List:
        weights associated with each gauss point
    """

    if dim != 2:
        raise NotImplementedError

    pts, wts = [], []
    
    if order == 1:
        pts.extend([0.0])
        wts.extend([2.0])
    elif order == 2:
        pts.extend([0.577350269189626, -0.577350269189626])
        wts.extend([1.0, 1.0])
    elif order == 3:
        pts.extend([0.774596669241483, -0.774596669241483, 0.0])
        wts.extend([0.555555555555556, 0.555555555555556, 0.888888888888889])
    else:
        raise NotImplementedError

    points, weights = [], []

    for i in range(order):
        for j in range(order):
            points.append([pts[i], pts[j]])
            weights.append(wts[i]* wts[j])
    points = np.array(points)
    return points, weights

if __name__ == '__main__':
    query_point = [-0.25, -0.25]
    shape_fns, derivatives = lagrange_basis('Q8', query_point)
    print("shape functions evaluated at ({},{}) are: ".format(query_point[0], 
                                                        query_point[1]))
    print(shape_fns)
    assert np.sum(shape_fns, axis=0) == 1.0, "shape functions should add up to unity"

    points, weights = gauss_quadrature(2)
    print('gauss points: ')
    print(points)
    print('associated weights: ')
    print(weights)