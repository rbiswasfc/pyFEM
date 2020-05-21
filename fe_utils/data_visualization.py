import os
import sys 
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class VisualizationModule(object): 
    """
    class to implement visualization functionality for different kind of plots
    """
    def __init__(self, config, df_nodes, df_elements):
        """
        initialize the visualization module

        @params:
        config -> dict:
            config dict for current analysis
        df_nodes -> pd.DataFrame:
            dataframe containing node information of current analysis
        df_elements -> pd.DataFrame:
            dataframe containing element information of current analysis
        """
        self.config = config
        self.output_dir = config["output_dir"]
        
        # create output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.node_data = df_nodes
        self.element_data = df_elements
        self.num_nodes = len(df_nodes)
        self.num_elem = len(df_elements)

    def find_node_along_axes(self, val = 0.0, direction = 'x', rel_tol = 1e-4):
        """
        function to find nodes in the axial directions
        @params:
        val -> float:
            nodes will be searched along x = val or y= val
        direction -> str:
            axial direction along which nodes will be searched
        rel_tol -> float:
            relative tolerance to match coordinates
        :return
        nodes -> List:
            list of nodes that satisfies the condition set by 
            val and direction parameters
        """
        valid_dirs = ['x', 'y']
        assert direction in valid_dirs

        df_tmp = self.node_data.copy()

        if direction == 'y':
            tol = rel_tol * df_tmp['coord1'].abs().max()
            df_tmp['flag'] = df_tmp['coord1'].apply(lambda x: abs(x-val) <tol)
            nodes = df_tmp[df_tmp['flag']].node_id.tolist()
        else:
            tol = rel_tol * df_tmp['coord2'].abs().max()
            df_tmp['flag'] = df_tmp['coord2'].apply(lambda x: abs(x-val)<tol)
            nodes = df_tmp[df_tmp['flag']].node_id.tolist()
        return nodes

    def get_nodeset_coords(self, node_list):
        """
        extract coordinates of a node set
        @param:
        node_list -> List:
            list of nodes in the node set
        """
        xs = self.node_data[self.node_data['node_id'].isin(node_list)].coord1.values
        ys = self.node_data[self.node_data['node_id'].isin(node_list)].coord2.values
        return xs, ys
    
    def plot_mesh(self, elem_type = 'Q8', cc = 'k-', save_fig = True, clear_fig = True):
        
        """
        plots the domain mesh and (optionally) saves the plot in output directory
        @param:
        elem_type -> str:
            type of element used for meshing
        cc -> str:
            color code and plot style
        save_fig -> boolean:
            flag for saving mesh plot
        clear_fig -> boolean:
            flag for clearing the current plot
        """
        if elem_type == 'Q8':
            pattern = [0,4,1,5,2,6,3,7,0] # for making edges
        else:
            raise NotImplementedError
        fig, ax = plt.subplots()

        for e in range(self.num_elem):
            elem = self.element_data.iloc[e]
            el_nodes = elem.values[1:].tolist()
            xpt, ypt = [], []
            for idx in pattern:
                this_node = self.node_data[self.node_data['node_id'] == el_nodes[idx]]
                cur_x = this_node.coord1.values[0]
                cur_y = this_node.coord2.values[0]
                xpt.append(cur_x)
                ypt.append(cur_y)

            plt.plot(xpt, ypt, cc)
        ax.set_title("Mesh Plot")
        if save_fig:
            save_path = os.path.join(self.config["output_dir"], 'mesh_plot.png')
            fig.savefig(save_path, dpi=200)
        if clear_fig:
            plt.clf()

    
    def plot_domain_boundary(self, save_fig = True):
        """
        plot and save mesh with domain boundaries
        @param:
        save_fig -> boolean:
            flag for saving the current figure
        """
        L, R = self.node_data['coord1'].min(), self.node_data['coord1'].max()
        B, T = self.node_data['coord2'].min(), self.node_data['coord2'].max()

        left_nodes = self.find_node_along_axes(val = L, direction = 'y')
        right_nodes = self.find_node_along_axes(val = R, direction = 'y')
        bot_nodes = self.find_node_along_axes(val = B, direction = 'x')
        top_nodes = self.find_node_along_axes(val = T, direction = 'x')

        # plot mesh
        self.plot_mesh(save_fig = False, clear_fig = False)
        
        # plot left boundary
        x, y = self.get_nodeset_coords(left_nodes)
        plt.scatter(x,y, color = 'red')

        # plot right boundary
        x, y = self.get_nodeset_coords(right_nodes)
        plt.scatter(x,y, color = 'blue')

        # plot bottom boundary
        x, y = self.get_nodeset_coords(bot_nodes)
        plt.scatter(x,y, color = 'green')

        # plot top boundary
        x, y = self.get_nodeset_coords(top_nodes)
        plt.scatter(x,y, color = 'cyan')

        if save_fig:
            save_path = os.path.join(self.config["output_dir"], 'mesh_plot_with_boundary.png')
            plt.savefig(save_path, dpi=200)
        plt.clf()

    
    def plot_mesh_with_gps(self, domain):
        pass

    def plot_disp_field(self, domain, step):
        pass
    
    def plot_deformed_geometry(self, domain, step):
        pass

    def plot_strain_field(self, domain, step):
        pass

    def plot_stress_field(self, domain, step):
        pass

if __name__ == "__main__":
    #from fe_utils.data_loader import MeshDataLoader
    from data_loader import MeshDataLoader
    filepath = './datasets/mesh_data.xlsx'
    sheet = 'homogenized_localization'
    data_loader = MeshDataLoader(filepath, sheet)
    df_nodes = data_loader.get_node_data('C10', 'D157')
    df_elements = data_loader.get_element_data('B159', 'I197')

    
    # print(os.getcwd())
    with open("./config.json", 'r') as f:
        config = json.load(f)
    

    data_visualiser = VisualizationModule(config, df_nodes, df_elements)
    data_visualiser.plot_mesh()
    data_visualiser.plot_domain_boundary()
    
    