import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from shapely.geometry import LineString, Polygon

import SearchAlgo
from Solver_Minimize import TrajectoryOptimizer

class PRM:
    def __init__(self, num_nodes, distance_radius, space_limits, obstacles_map=None):
        self.num_nodes = num_nodes
        self.distance_radius = distance_radius
        self.space_limits = space_limits
        self.theta_diff = np.pi / 6
        self.max_stirring = np.pi / 3
        self.max_dist_error = 0.05
        self.obstacles = obstacles_map
        self.solver = TrajectoryOptimizer()

        self.shortest_path = None
        self.nodes = self.generate_random_nodes()
        self.edges = self.generate_edges()

    def generate_random_nodes(self):
        nodes = np.random.rand(self.num_nodes, 3)  # (x, y, theta)
        nodes[:, 0] = nodes[:, 0] * (self.space_limits[0][1] - self.space_limits[0][0]) + self.space_limits[0][0]  # scale x
        nodes[:, 1] = nodes[:, 1] * (self.space_limits[1][1] - self.space_limits[1][0]) + self.space_limits[1][0]  # scale y
        nodes[:, 2] = nodes[:, 2] * (2*np.pi)  # angle in radians
        return nodes

    def theta_distance(self, theta1, theta2):
        diff = abs(theta1 - theta2) % (2 * np.pi)
        return 2 * np.pi - diff if diff > np.pi else diff

    def obstacles_intersection(self, node1, node2):
        line = LineString([(node1[0], node1[1]), (node2[0], node2[1])])

        for obstacle in self.obstacles:
            if obstacle.intersects(line):
                return False
        return True

    def generate_edges(self):
        tree = KDTree(self.nodes[:, :2], leafsize=15)  # KDTree for efficient nearest neighbor search
        edges = set()  # Use set to ensure uniqueness
        for i in range(self.num_nodes):
            self.generate_edges_curve_single_node(tree, edges, i)
            # Query nodes within distance radius
            
        return list(edges)
    
    # def generate_edges_straight_single_node(self, tree, edges, node_index, check_angle=True):
    #     indices = tree.query_ball_point(self.nodes[node_index, :2], self.distance_radius)
    #     for j in indices:
    #         if node_index != j:
    #             theta1 = self.nodes[node_index, 2]
    #             theta2 = self.nodes[j, 2]
    #             theta_diff = self.theta_distance(theta1, theta2)
    #             if (not check_angle or theta_diff <= self.theta_diff) and (j,node_index) not in edges and self.obstacles_intersection(self.nodes[node_index], self.nodes[j]):
    #                 edges.add((node_index, j))
    #     pass
    
    def generate_edges_curve_single_node(self, tree, edges, node_index, check_angle=True):
        indices = tree.query_ball_point(self.nodes[node_index, :2], self.distance_radius)
        for j in indices:
            if node_index != j:
                theta1 = self.nodes[node_index, 2]
                theta2 = self.nodes[j, 2]
                theta_diff = self.theta_distance(theta1, theta2)
                if (not check_angle or theta_diff <= self.theta_diff) and (j,node_index) not in edges and self.obstacles_intersection(self.nodes[node_index], self.nodes[j]):
                    # first direction node_index -> j
                    opt_v, opt_delta = self.solver.solve(self.nodes[node_index], self.nodes[j])
                    if (opt_v>0 and np.abs(opt_delta) < self.max_stirring):
                        dest = self.solver.destination(opt_v, opt_delta,self.nodes[node_index] )
                        slice_range = self.solver.slice_range(self.nodes[node_index][2], dest[2], opt_delta)
                        if ((dest[0] - self.nodes[j][0])**2 + (dest[1] - self.nodes[j][1])**2) < self.max_dist_error:
                            edges.add((node_index, j, opt_v, opt_delta, slice_range))
                        
                    # second direction j -> node_index
                    opt_v, opt_delta = self.solver.solve(self.nodes[j], self.nodes[node_index])
                    if (opt_v>0 and np.abs(opt_delta) < self.max_stirring):
                        dest = self.solver.destination(opt_v, opt_delta,self.nodes[j] )
                        slice_range = self.solver.slice_range(self.nodes[j][2], dest[2], opt_delta)
                        if ((dest[0] - self.nodes[node_index][0])**2 + (dest[1] - self.nodes[node_index][1])**2) < self.max_dist_error:
                            edges.add((j, node_index, opt_v, opt_delta, slice_range))
        pass
    
    def FindRoadMap(self, start_node, end_node, searchAlg='Dijkstra'):
        self.nodes = np.concatenate((self.nodes, np.reshape(start_node, (1,3)), np.reshape(end_node, (1,3))), axis=0)
        start_index = len(self.nodes) - 2
        end_index = len(self.nodes) - 1

        tree = KDTree(self.nodes[:, :2], leafsize=15)
        edges = set(self.edges)

        self.generate_edges_curve_single_node(tree, edges, start_index, check_angle=False)
        self.generate_edges_curve_single_node(tree, edges, end_index, check_angle=False)

        #TODO maybe add multiple search functions
        if searchAlg == 'Dijkstra':
            shortest_path, _ = SearchAlgo.Dijkstra(self.nodes, edges, start_index, end_index)
        elif searchAlg == 'A_Star':
            pass
        else:
            assert(0)  
        
        if len(shortest_path) > 0:
            a = np.array(shortest_path, dtype=int)[:, 0]
            b = np.array(shortest_path, dtype=int)[:, 1]
            c = np.array(shortest_path)[:, 2:]
            self.shortest_path = np.concatenate((self.nodes[a], self.nodes[b], c), axis=1).tolist()
        else:
            self.shortest_path = shortest_path
        # remove start and end nodes at the end.
    

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1], c=self.nodes[:, 2], cmap='hsv', label='Nodes')

        for edge in self.edges:
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='black', alpha=0.3)

        for obstacle in self.obstacles:
            x,y = obstacle.exterior.xy
            plt.plot(x,y, c='black')

        if self.shortest_path is not None:
            for i in range(np.size(self.shortest_path, axis=0)):
                start_node = self.shortest_path[i][0:3]
                goal_node = self.shortest_path[i][3:6]
                v = self.shortest_path[i][6]
                phi = self.shortest_path[i][7]
                # plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='red', alpha=1)
                self.solver.plot_trajectory(v, phi, start_node, goal_node)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PRM')
        # plt.legend()
        plt.grid(True)
        plt.colorbar(label='Theta (radians)')
        plt.show()

def generate_obstacles():
    # Create a list to store polygons representing obstacles
    obstacles = []
    
    # Define vertices for walls as polygons
    obstacles.append(Polygon([(0, 90), (0, 85), (80, 85), (80, 90)]))
    obstacles.append(Polygon([(100, 65), (100, 60), (20, 60), (20, 65)]))
    obstacles.append(Polygon([(0, 40), (0, 35), (80, 35), (80, 40)]))
    obstacles.append(Polygon([(100, 20), (100, 15), (20, 15), (20, 20)]))

    return obstacles

if __name__ == "__main__":
    # Constants
    num_nodes = 500
    distance_radius = 20  # Adjust distance radius as needed
    space_limits = [(0, 100), (0, 100)]  # Limits for x and y coordinates

    # Obstacles
    obstacle_polygons = [
        Polygon([(20, 20), (80, 20), (80, 25), (20,25)]),
        Polygon([(20, 60), (80, 70), (50, 80)])
                        ]
    # obstacle_polygons = generate_obstacles()
    #obstacle_polygons = []

    # Test
    prm = PRM(num_nodes, distance_radius, space_limits, obstacle_polygons)
    prm.FindRoadMap([10,20,0], [90,90,0], 'Dijkstra')
    # prm.FindRoadMap([90,5,0], [10,95,0], 'A_Star')
    prm.plot()
    a = 0
    