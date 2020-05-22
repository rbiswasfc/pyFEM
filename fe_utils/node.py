import os
import sys
import numpy as np
import pandas as pd
from operator import add


class Node(object):
    """
    class to provide abstraction for nodes in FE Analysis
    """

    def __init__(self, node_id, org_coords):
        """
        initialize the node object with original coordinates and displacement field
        @params:
        node_id -> Int:
            node number of this node
        org_coords -> List:
            original coordinates of this node
        """
        assert len(org_coords) == 2, "only 2-D analysis supported"
        self.org_coords = org_coords
        self.disp = [0.0, 0.0]

    @property
    def cur_coords(self):
        """
        get current coordinates of the node
        """
        current_coords = list(map(add, self.org_coords, self.disp))
        return current_coords

if __name__ == '__main__':
    node = Node(0, [1.8, 2.7])
    node.disp = [0.2, 0.3]
    print(node.org_coords)
    print(node.cur_coords)