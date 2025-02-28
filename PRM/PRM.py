import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from shapely.geometry import LineString, Polygon

from .SearchAlgo import *
from .Solver_Minimize import TrajectoryOptimizer
from scripts import utils

class PRM:
    def __init__(self, num_nodes, distance_radius, space_limits, start_point, end_point, obstacles_map=None, seed=None, model=None):
        self.num_nodes = num_nodes
        self.distance_radius = distance_radius
        self.space_limits = space_limits
        self.obstacles = obstacles_map
        
        #limitation on road:
        self.theta_diff_before = 180 * np.pi/180  # before finding solution
        self.theta_diff_after = 60 * np.pi/180   # after finding solution
        self.max_stirring = np.pi/2
        self.max_dist_error = 0.05 #in meter

        self.start_point = start_point
        self.end_point = end_point

        #solver's data:
        self.solver = TrajectoryOptimizer(L=2, distance_radius=self.distance_radius)
        self.shortest_path = None
        
        self.nodes = self.generate_random_nodes(seed)
        
        self.model = model
        if self.model:
            self.edges = self.generate_edge_mlp()
        else:
            self.edges = self.generate_edges()

    def generate_random_nodes(self, seed=None):
        print("-- Generating Nodes --")
        if seed is not None:
            np.random.seed(seed)

        nodes = np.random.rand(self.num_nodes, 3)  # (x, y, theta)
        nodes[:, 0] = nodes[:, 0] * (self.space_limits[0][1] - self.space_limits[0][0]) + self.space_limits[0][0]
        nodes[:, 1] = nodes[:, 1] * (self.space_limits[1][1] - self.space_limits[1][0]) + self.space_limits[1][0]
        nodes[:, 2] = nodes[:, 2] * (2*np.pi) 
        nodes = np.concatenate((nodes, np.reshape(self.start_point, (1,3)), np.reshape(self.end_point, (1,3))), axis=0)
        
        np.random.seed(None)
        return nodes

    def obstacles_trajectory_intersection(self, trajectory: list):
        x_traj, y_traj, _ = trajectory
        intersects = np.zeros(len(x_traj), dtype=bool)
        
        for i in range(len(x_traj)):
            line = LineString(np.column_stack((x_traj[i], y_traj[i])))
            for obstacle in self.obstacles:
                if obstacle.intersects(line):
                    intersects[i] = True

        return intersects

    def oclidian_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def limit_by_distance(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2) < self.max_dist_error
    
    def limit_by_theta(self, theta1, theta2, theta_error):
        return np.abs(theta1 - theta2) < theta_error
    
    def limit_by_velocity_stirring_time(self, v, stirring, T):
        return ( (np.array(v) > 0) & (np.abs(np.array(stirring)) < self.max_stirring) & (T > 0))
    
    def translate_rotate(self, start, goal):
        start = np.array(start)
        goal = np.array(goal)
        
        x_start, y_start, theta_start = start
        x_goal, y_goal, theta_goal = goal.T

        x_goal_translated = x_goal - x_start
        y_goal_translated = y_goal - y_start
        
        R = np.array([[np.cos(-theta_start), -np.sin(-theta_start)],
                  [np.sin(-theta_start), np.cos(-theta_start)]])

        temp = np.array([x_goal_translated, y_goal_translated])
        goal_rotated = np.dot(R, temp)
        
        return np.array([*goal_rotated, theta_goal - theta_start]).T

    # Used only for model based solution
    def generate_edge_mlp(self):
        print("-- Generating Edges --")
        
        tree = KDTree(self.nodes[:, :2])  # KDTree for efficient nearest neighbor search
        edges = set()  # Use set to ensure uniqueness
        for begin_indx in range(self.num_nodes+2):
            begin_node = self.nodes[begin_indx, :]
            indices = np.array(tree.query_ball_point(self.nodes[begin_indx, :2], self.distance_radius), dtype=int)
            indices = indices[indices != begin_indx]
            if len(indices) == 0: continue
            neighbors = self.nodes[indices, :]
            
            theta_limited_before = self.limit_by_theta(begin_node[2], neighbors[:, 2], self.theta_diff_before)
            indices = indices[theta_limited_before]
            if len(indices) == 0: continue
            neighbors = self.nodes[indices, :]
            
            deltas = self.translate_rotate(begin_node, neighbors)
            y = self.model(torch.tensor(deltas))
            v, stir, T = np.array(y.T.squeeze().tolist())
            
            solution_limit = self.limit_by_velocity_stirring_time(v, stir, T)
            indices = indices[solution_limit]
            if len(indices) == 0: continue
            neighbors = self.nodes[indices, :]
            v, stir, T = v[solution_limit], stir[solution_limit], T[solution_limit]
            
            trajectory, T = utils.get_trajectory(v, stir, T, begin_node, neighbors, self.solver.L)
            trajectory, _ = utils.get_trajectory(v, stir, T, begin_node, neighbors, self.solver.L)
            
            theta_limited_after = self.limit_by_theta(trajectory[2][:, -1], neighbors[:, 2], self.theta_diff_after)
            limit_dist = self.limit_by_distance(neighbors[:, 0], neighbors[:, 1], trajectory[0][:, -1], trajectory[1][:, -1])
            filter = theta_limited_after & limit_dist
            indices = indices[filter]
            if len(indices) == 0: continue
            trajectory = (trajectory[0][filter, :], trajectory[1][filter, :], trajectory[2][filter, :])
            neighbors = self.nodes[indices, :]
            v, stir, T = v[filter], stir[filter], T[filter]
            
            object_intersects = self.obstacles_trajectory_intersection(trajectory)
            indices = indices[~object_intersects]
            if len(indices) == 0: continue
            trajectory = (trajectory[0][~object_intersects, :], trajectory[1][~object_intersects, :], trajectory[2][~object_intersects, :])
            neighbors = self.nodes[indices, :]
            v, stir, T = v[~object_intersects], stir[~object_intersects], T[~object_intersects]
            
            dest = utils.destination(self.solver.L, T, v, stir, begin_node)
            weight = self.solver.edge_road_weight(begin_node[2], dest[2], stir)
            
            for j in range(len(indices)):
                edges.add((begin_indx, indices[j], v[j], stir[j], weight[j], T[j]))
          
        return edges  

    def generate_edges(self):
        print("-- Generating Edges --")

        tree = KDTree(self.nodes[:, :2])  # KDTree for efficient nearest neighbor search
        edges = set()  # Use set to ensure uniqueness
        for i in range(self.num_nodes+2):
            self.generate_edges_curve_single_node(tree, edges, i)     
        return edges

    # Used only for ODE solver
    # Generate all solutions between node_index to all neighbors within distance_radius
    def generate_edges_curve_single_node(self, tree: KDTree, edges: set, node_index: int):
        indices = tree.query_ball_point(self.nodes[node_index, :2], self.distance_radius)
        for idx in indices:
            if node_index != idx:
                begin_node = self.nodes[node_index]
                end_node = self.nodes[idx]
                T_calc = self.oclidian_distance(begin_node[0], begin_node[1], end_node[0], end_node[1]) / 10
                if (T_calc >8): T_calc = 8
                if (T_calc<1): T_calc = 1
                self.solver.T = T_calc
                self.solver.dt = T_calc / 100
                
                theta_limited = self.limit_by_theta(begin_node[2], end_node[2],self.theta_diff_before)
                if ( theta_limited ):
                    v, stir = self.solver.solve(begin_node, end_node)
                    self.solver.dt = self.solver.T / 100
                    
                    solution_limit = self.limit_by_velocity_stirring_time(v, stir, self.solver.T)
                    if (solution_limit):
                        trajectory, _ = utils.get_trajectory(v, stir, self.solver.T, begin_node, end_node, self.solver.L)
                        dest = utils.destination(self.solver.L, self.solver.T, v, stir, begin_node)
                        weight = self.solver.edge_road_weight(begin_node[2], dest[2], stir)

                        theta_limited = self.limit_by_theta(trajectory[2][0, -1], end_node[2],self.theta_diff_after)
                        limit_good = self.limit_by_distance(end_node[0], end_node[1], trajectory[0][0, -1], trajectory[1][0, -1])

                        if (limit_good and theta_limited and not (self.obstacles_trajectory_intersection(trajectory) ) ):
                            edges.add((node_index, idx, v, stir, weight, self.solver.T))


    def FindRoadMap(self, searchAlg='Dijkstra'):
        print("-- Finding Solution --")

        start_index = len(self.nodes) - 2
        end_index = len(self.nodes) - 1
        edges = self.edges

        if searchAlg == 'Dijkstra':
            shortest_path, path_length = Dijkstra(self.nodes, edges, start_index, end_index)
        else:
            assert(0)  
        
        if len(shortest_path) > 0:
            a = np.array(shortest_path, dtype=int)[:, 0]
            b = np.array(shortest_path, dtype=int)[:, 1]
            c = np.array(shortest_path)[:, 2:]
            self.shortest_path = np.concatenate((self.nodes[a], self.nodes[b], c), axis=1).tolist()
        else:
            self.shortest_path = shortest_path

        return path_length

    def plot(self):
        f = plt.figure(figsize=(8, 8))
        
        # Plot nodes
        plt.quiver(self.nodes[:, 0], self.nodes[:, 1], np.cos(self.nodes[:, 2]), np.sin(self.nodes[:, 2]), 
                   scale=100, width=0.002, color='b', label='_Hidden label', zorder=1)

        # Plot edges as straight lines
        for edge in self.edges:
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='black', alpha=0.3, zorder=2)

        # Plot obstacles
        for obstacle in self.obstacles:
            x,y = obstacle.exterior.xy
            plt.plot(x,y, c='black', zorder=3)

        # Plot path as curved lines
        if self.shortest_path is not None:
            for i in range(np.size(self.shortest_path, axis=0)):
                start_node = self.shortest_path[i][0:3]
                v = self.shortest_path[i][6]
                phi = self.shortest_path[i][7]
                T = self.shortest_path[i][9]
                utils.plot_trajectory(v, phi, T, start_node, self.solver.L, 'r')

        # Plot start point
        plt.quiver(self.nodes[-2, 0], self.nodes[-2, 1], np.cos(self.nodes[-2, 2]), np.sin(self.nodes[-2, 2]), 
                   scale=100, width=0.002, color='g', label='Start', zorder=4)
        # Plot end point
        plt.quiver(self.nodes[-1, 0], self.nodes[-1, 1], np.cos(self.nodes[-1, 2]), np.sin(self.nodes[-1, 2]), 
                   scale=100, width=0.002, color='g', label='End', zorder=4)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PRM')
        plt.grid(True)

        return f

    def plotAsSingles(self):
            if self.shortest_path is not None:
                for i in range(np.size(self.shortest_path, axis=0)):
                    start_node = self.shortest_path[i][0:3]
                    goal_node = self.shortest_path[i][3:6]
                    v = self.shortest_path[i][6]
                    phi = self.shortest_path[i][7]
                    T = self.shortest_path[i][9]
                    plt.figure(figsize=(8, 8))
                    
                    plt.xlim(min(start_node[0], goal_node[0]) - 10, max(start_node[0], goal_node[0]) + 10)
                    plt.ylim(min(start_node[1], goal_node[1]) - 10, max(start_node[1], goal_node[1]) + 10)

                    plt.scatter(goal_node[0], goal_node[1], color='green', label='Goal', zorder=5)
                    plt.scatter(start_node[0], start_node[1], color='green', label='Initial', zorder=5)
                    utils.plot_trajectory(v, phi, T, start_node, self.solver.L)
                    plt.quiver(goal_node[0], goal_node[1], np.cos(goal_node[2]), np.sin(goal_node[2]), scale=10, color='b', label='_Hidden label')
                    plt.quiver(start_node[0], start_node[1], np.cos(start_node[2]), np.sin(start_node[2]), scale=10, color='b', label='_Hidden label')
