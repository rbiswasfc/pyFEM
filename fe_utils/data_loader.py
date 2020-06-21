import os
import sys

import numpy as np
import pandas as pd

from openpyxl import load_workbook


class MeshDataLoader(object):
    '''
    class to read Abaqus generated input file 
    '''

    def __init__(self, filepath, sheet='Sheet1'):
        '''
        initialize the MashDataLoader class

        @params:
        filepath -> str:
          path to the excel file containing mesh information
        sheet -> str:
          specify the sheet that contains mesh information, defaults to Sheet1
        '''
        self.filepath = filepath
        self.sheet = sheet

    def _read_data(self, start, end):
        '''
        private function to facilitate reading of data from excel file

        @params:
        start -> str:
          start cell to begin reading e.g. 'A1'
        end -> str:
          end cell to stop reading e.g. 'B10'

        @returns:
        data_rows -> List[List]
          List containing list of nodes/elements 
        '''

        try:
            data_obj = load_workbook(filename=self.filepath, read_only=True)
        except:
            print("Warning! {} is missing! No data is read!".format(self.filename))
            return

        try:
            data = data_obj[self.sheet]
        except:
            print("Warning! Sheet {} is missing! No data is read!".format(self.sheet))
            return

        data_rows = []
        for row in data[start:end]:
            data_cols = []
            for cell in row:
                data_cols.append(cell.value)
            data_rows.append(data_cols)
        return data_rows

    def get_node_data(self, start, end):
      
      '''
      Function to read in node data for analysis
      Note: In Abaqus generated input files node numbering starts from 1
      Here choice is made to start node id from zero for convenience.
      This function reads in node data and handles the conversion.

      @params:
      start -> str:
        start cell to begin reading e.g. 'A1'
      end -> str:
        end cell to stop reading e.g. 'B10'

      @returns:
      df_nodes -> pd.DataFrame:
        dataframe containing node data with columns node_id, coord1 and coord2
      '''
      
      nodes = self._read_data(start, end)
      df_nodes = pd.DataFrame(nodes)
      assert df_nodes.shape[1] == 2, "Only 2-D is supported"
      df_nodes.columns = ['coord1', 'coord2']
      df_nodes = df_nodes.reset_index()
      df_nodes['node_id'] = df_nodes['index']
      df_nodes = df_nodes[['node_id', 'coord1', 'coord2']].copy()

      return df_nodes

    def get_element_data(self, start, end):
      '''
      Function to read in element data for analysis
      Note: In Abaqus generated input files element numbering starts from 1
      Here choice is made to start elem id from zero for convenience.
      This function reads in element data and handles the conversion.

      @params:
      start -> str:
        start cell to begin reading e.g. 'A1'
      end -> str:
        end cell to stop reading e.g. 'B10'

      @returns:
      df_elements -> pd.DataFrame:
        dataframe containing element data with columns elem_id, n1, n2, ...
      '''
      elements = self._read_data(start, end)
      df_elements = pd.DataFrame(elements)
      assert df_elements.shape[1] == 8, "Only Q8 element is supported as of now"

      node_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
      df_elements.columns = node_cols
      df_elements = df_elements.reset_index()
      df_elements['elem_id'] = df_elements['index'] 
      for col in node_cols:
        # decrease by 1 to reflect node numbering starts from zero
        df_elements[col] = df_elements[col]-1 
      df_elements = df_elements[['elem_id', 'n1', 'n2',
                                  'n3', 'n4', 'n5', 'n6', 'n7', 'n8']].copy()
      df_elements = df_elements.astype(int)
      return df_elements


#############
if __name__ == '__main__':
    filepath = './datasets/mesh_data.xlsx'
    sheet = 'homogenized_localization'
    data_loader = MeshDataLoader(filepath, sheet)
    df_nodes = data_loader.get_node_data('C10', 'D157')
    df_elements = data_loader.get_element_data('B159', 'I197')
    print('=='*20)
    print('Nodes:')
    print(df_nodes.head())
    
    print('=='*20)
    print('Elements:')
    print(df_elements.head())



