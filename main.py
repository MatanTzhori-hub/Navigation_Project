from PRM.Solver_Minimize import *
from PRM.PRM import *
from shapely.geometry import LineString, Polygon, Point
import time
import datetime
import itertools
import os

distance_radius = 15
space_limits = [(0, 50), (0, 50)] 

seed = 42

# Obstacles
map_0 = []
map_1 = [ Polygon([(0.0, 15), (0, 18), (35, 18), (35, 15)]),  ## 2 barrier road
          Polygon([(50, 35), (50, 32.0), (15, 32), (15, 35)]),
          Polygon([(0.0, 0.0), (1, 0.0), (1, 50), (0.0, 50)]),
          Polygon([(49, 0.0), (50, 0.0), (50, 50), (49, 50)])]
map_2 = [
    Polygon([(10.0, 10.0), (40.0, 10.0), (40.0, 12.5), (10.0, 12.5)]),
    Polygon([(10.0, 30.0), (40.0, 35.0), (25.0, 40.0)])]
map_3 = [
    Polygon([(15.0, 15.0), (30.0, 15.0), (22.5, 30.0)]), ## Magen David
    Polygon([(15.0, 25.0), (30.0, 25.0), (22.5, 10.0)])]
map_4 = [ Polygon([(10, 10), (12, 10), (12, 20), (10, 20)]),  ## 5 brariers
          Polygon([(10, 40), (12, 40), (12, 30), (10, 30)]),
          Polygon([(38, 10), (40, 10), (40, 20), (38, 20)]),
          Polygon([(38, 40), (40, 40), (40, 30), (38, 30)]),
          Polygon([(24, 16), (26, 16), (26, 34), (24, 34)]),
          Polygon([(0, 0), (50, 0), (50, -1), (0, -1)]),
          Polygon([(0, 50), (50, 50), (50, 51), (0, 51)])]


# Setup
maps = [map_0, map_1, map_2, map_3, map_4]
models_names = ['MLP_short', 'MLP_deep', 'MOE_thin', 'MOE_wide']
models = {"MLP_short": torch.load("checkpoints/model_MLP_[3, 128, 3]__12_21_17_34_22"),
          "MLP_deep": torch.load("checkpoints/model_MLP_[3, 256, 256, 256, 256, 256, 256, 256, 256, 3]__12_22_03_08_06"),
          "MOE_thin":torch.load("checkpoints/model_MOE_[3, 128, 128, 128, 3]_E5K2__12_20_11_01_03"),
          "MOE_wide":torch.load("checkpoints/model_MOE_[3, 128, 128, 128, 128, 3]_E13K6__12_20_13_24_23")
          }
num_nodes = [500, 1000, 2000, 2500, 4000]
num_nodes_solver = [50, 100, 200, 500]


end_point = [45, 45, 0]
start_point = [5, 5, 0]


# Loop
solver_pool = itertools.product(maps, num_nodes_solver)
model_pool = itertools.product(maps, models_names, num_nodes)

date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
os.mkdir(f"figures/{date}")

for combo in solver_pool:
    t_start =  time.time()
    map, n_nodes = combo
    prm_solver = PRM(n_nodes, distance_radius, space_limits, start_point, end_point, map, seed=seed, model=None)
    prm_solver.FindRoadMap('Dijkstra')
    t_end =  time.time()
    
    f = prm_solver.plot()
    plt.title(f"""Elapsed time: {t_end-t_start:.2f} 
            Number of Nodes: {prm_solver.num_nodes}, Connect Radius: {prm_solver.distance_radius}, 
            Theta diff before: {prm_solver.theta_diff_before*(180/np.pi):.1f}, Theta diff after: {prm_solver.theta_diff_after*(180/np.pi):.1f}, 
            Max distance error: {prm_solver.max_dist_error}""")
    f.savefig(f"figures/{date}/Solver_{t_start}.png", dpi=500)
    plt.close()

for combo in model_pool:
    t_start =  time.time()
    map, model_name, n_nodes = combo
    model = models[model_name]
    prm_mlp = PRM(n_nodes, distance_radius, space_limits,start_point,end_point, map, seed=seed, model=model)
    prm_mlp.FindRoadMap('Dijkstra')
    t_end =  time.time()

    f = prm_mlp.plot()
    plt.title(f"""Elapsed time: {t_end-t_start:.2f} 
                Number of Nodes: {prm_mlp.num_nodes}, Connect Radius: {prm_mlp.distance_radius}, 
                Theta diff before: {prm_mlp.theta_diff_before*(180/np.pi):.1f}, Theta diff after: {prm_mlp.theta_diff_after*(180/np.pi):.1f}, 
                Max distance error: {prm_mlp.max_dist_error}""")
    f.savefig(f"figures/{date}/Model_{model_name}_{t_start}.png", dpi=500)
    plt.close()
