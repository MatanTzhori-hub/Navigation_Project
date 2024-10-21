import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from shapely.geometry import LineString, Polygon, Point

from .SearchAlgo import *
from .Solver_Minimize import TrajectoryOptimizer
from scripts import utils

class PRM:
    def __init__(self, num_nodes, distance_radius, space_limits, start_point,end_point,obstacles_map=None, seed=None):
        self.num_nodes = num_nodes
        self.distance_radius = distance_radius
        self.space_limits = space_limits
        self.obstacles = obstacles_map
        #limitation on road:
        self.theta_diff_before = 135 * np.pi/180  # before finding solution
        self.theta_diff_after = 60 * np.pi/180   # after finding solution
        self.max_stirring = np.pi/2
        self.max_dist_error =0.05 #in meter

        self.start_point = start_point
        self.end_point = end_point

        #solver's data:
        self.solver = TrajectoryOptimizer(distance_radius=self.distance_radius)
        self.shortest_path = None

        self.nodes = self.generate_random_nodes(seed)
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

    # def theta_distance(self, theta1, theta2):
    #     diff = abs(theta1 - theta2) % (2 * np.pi)
    #     return 2 * np.pi - diff if diff > np.pi else diff
    
    # def check_theta_limitation(self, node1, node2):
    #     x1, y1, theta1 = node1
    #     x2, y2, theta2 = node2
    #     theta3 = self.angle_between_points((x1, y1),(x2, y2))
    #     return self.theta_distance(theta1, theta3) < self.theta_diff and self.theta_distance(theta2, theta3) < self.theta_diff
        
    # def angle_between_points(self, p1, p2):
    #     x1, y1 = p1
    #     x2, y2 = p2
    #     dot_product = x1 * x2 + y1 * y2
    #     magnitude_p1 = np.sqrt(x1**2 + y1**2)
    #     magnitude_p2 = np.sqrt(x2**2 + y2**2)
    #     cos_theta = dot_product / (magnitude_p1 * magnitude_p2)
    #     cos_theta = np.clip(cos_theta, -1.0, 1.0)
    #     theta = np.arccos(cos_theta)
    #     return theta
        

    def obstacles_line_intersection(self, node1, node2):
        line = LineString([(node1[0], node1[1]), (node2[0], node2[1])])
        for obstacle in self.obstacles:
            if obstacle.intersects(line):
                return True
        return False
    
    def obstacles_trajectory_intersection(self, trajectory: list):
        for obstacle in self.obstacles:
            for point in zip(trajectory[0][::10], trajectory[1][::10]):
                if obstacle.intersects(Point(point)):
                    return True
        return False

    def oclidian_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def limit_by_distance(self, x1, y1, x2, y2):
        if ((x1 - x2)**2 + (y1 - y2)**2) < self.max_dist_error:
            return True
        return False
    
    def limit_by_theta(self, theta1, theta2, theta_error):
        if np.abs(theta1 - theta2) < theta_error:
            return True
        return False
    
    def limit_by_velocity_stirring(self, v, stirring):
        if ( v > 0 and np.abs(stirring) < self.max_stirring):
            return True
        return False


    def generate_edges(self):
        print("-- Generating Edges --")

        tree = KDTree(self.nodes[:, :2])  # KDTree for efficient nearest neighbor search
        edges = set()  # Use set to ensure uniqueness
        for i in range(self.num_nodes+2):
            self.generate_edges_curve_single_node(tree, edges, i)     
        return list(edges)

    #generate all solutions between node_index to all neighbors within distance_radius
    def generate_edges_curve_single_node(self, tree, edges, node_index):
        indices = tree.query_ball_point(self.nodes[node_index, :2], self.distance_radius)
        for idx in indices:
            if node_index != idx:
                begin_node = self.nodes[node_index]
                end_node = self.nodes[idx]
                T_calc = self.oclidian_distance(begin_node[0], begin_node[1], end_node[0], end_node[1]) / 10
                if (T_calc >8): T_calc =8
                if (T_calc<1): T_calc = 1
                self.solver.T = T_calc
                
                theta_limited = self.limit_by_theta(begin_node[2], end_node[2],self.theta_diff_before)
                if ( theta_limited ):
                    v, stir = self.solver.solve(begin_node, end_node)
                    solution_limit = self.limit_by_velocity_stirring(v,stir)
                    if (solution_limit):
                        trajectory = self.solver.get_trajectory(v, stir, begin_node)
                        dest = utils.destination(self.solver.L, self.solver.T, v, stir,begin_node)
                        weight = self.solver.edge_road_weight(begin_node[2], dest[2], stir)

                        theta_limited = self.limit_by_theta(trajectory[2][-1], end_node[2],self.theta_diff_after)
                        # limit_good = any(self.limit_by_distance(x, y, end_node[0], end_node[1]) for x,y,_ in zip(*trajectory))
                        limit_good = self.limit_by_distance(end_node[0], end_node[1], trajectory[0][-1], trajectory[1][-1])

                        if (limit_good and theta_limited and not (self.obstacles_trajectory_intersection(trajectory) ) ):
                            edges.add((node_index, idx, v, stir, weight, self.solver.T))


    def FindRoadMap(self, start_node, end_node, searchAlg='Dijkstra'):
        print("-- Finding Solution --")

        #self.nodes = np.concatenate((self.nodes, np.reshape(start_node, (1,3)), np.reshape(end_node, (1,3))), axis=0)
        start_index = len(self.nodes) - 2
        end_index = len(self.nodes) - 1

        edges = set(self.edges)


        if searchAlg == 'Dijkstra':
            shortest_path, _ = Dijkstra(self.nodes, edges, start_index, end_index)
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
        f = plt.figure(figsize=(8, 8))
        plt.quiver(self.nodes[:, 0], self.nodes[:, 1], np.cos(self.nodes[:, 2]), np.sin(self.nodes[:, 2]), 
                   scale=100, width=0.002, color='b', label='_Hidden label')

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
                self.solver.T = self.shortest_path[i][9]
                self.solver.plot_trajectory(v, phi, start_node)

        # Plot start point
        plt.quiver(self.nodes[-2, 0], self.nodes[-2, 1], np.cos(self.nodes[-2, 2]), np.sin(self.nodes[-2, 2]), 
                   scale=100, width=0.002, color='g', label='Start')
        # Plot end point
        plt.quiver(self.nodes[-1, 0], self.nodes[-1, 1], np.cos(self.nodes[-1, 2]), np.sin(self.nodes[-1, 2]), 
                   scale=100, width=0.002, color='g', label='End')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PRM')
        # plt.legend()
        plt.grid(True)

        return f

    def plotAsSingles(self):
            if self.shortest_path is not None:
                for i in range(np.size(self.shortest_path, axis=0)):
                    start_node = self.shortest_path[i][0:3]
                    goal_node = self.shortest_path[i][3:6]
                    v = self.shortest_path[i][6]
                    phi = self.shortest_path[i][7]
                    plt.figure(figsize=(8, 8))
                    
                    plt.xlim(min(start_node[0], goal_node[0]) - 10, max(start_node[0], goal_node[0]) + 10)
                    plt.ylim(min(start_node[1], goal_node[1]) - 10, max(start_node[1], goal_node[1]) + 10)

                    plt.scatter(goal_node[0], goal_node[1], color='green', label='Goal', zorder=5)
                    plt.scatter(start_node[0], start_node[1], color='green', label='Initial', zorder=5)
                    self.solver.plot_trajectory(v, phi, start_node)
                    plt.quiver(goal_node[0], goal_node[1], np.cos(goal_node[2]), np.sin(goal_node[2]), scale=10, color='b', label='_Hidden label')
                    plt.quiver(start_node[0], start_node[1], np.cos(start_node[2]), np.sin(start_node[2]), scale=10, color='b', label='_Hidden label')
                    

            #plt.xlabel('X')
            #plt.ylabel('Y')
            #plt.title('PRM')
            # plt.legend()
            #plt.grid(True)
            # plt.colorbar(label='Theta (radians)')
            #plt.show()




if __name__ == "__main__":
    # Constants
    num_nodes = 500
    distance_radius = 30  # Adjust distance radius as needed
    space_limits = [(0, 100), (0, 100)]  # Limits for x and y coordinates

    # Obstacles
    obstacle_polygons = [
        Polygon([(20, 20), (80, 20), (80, 25), (20,25)]),
        Polygon([(20, 60), (80, 70), (50, 80)])]

    # Test
    prm = PRM(num_nodes, distance_radius, space_limits, obstacle_polygons)
    prm.FindRoadMap([10,20,0], [90,90,0], 'Dijkstra')
    # prm.FindRoadMap([90,5,0], [10,95,0], 'A_Star')
    prm.plot()
    plt.show()