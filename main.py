from PRM.Solver_Minimize import *
from PRM.PRM import *
from shapely.geometry import LineString, Polygon, Point
import time
import datetime

num_nodes = 5000
distance_radius = 20
space_limits = [(0, 100), (0, 100)] 

seed = 42

# Obstacles
map_0 = []

map_1 = [ Polygon([(70.0, 50.0), (60.0, 65.0), (40.0, 65.0), (30.0, 50.0), (40.0, 35.0), (60.0, 35.0)]) ]

map_2 = [ Polygon([(40, 40), (60, 60), (40, 60), (60,40)]) ]

map_3 = [ Polygon([(0, 30), (70, 30), (70, 35), (0, 35)]),
          Polygon([(100, 65), (30, 65), (30, 70), (100, 70)]),
          Polygon([(0, 0), (1, 0), (1, 100), (0, 100)]),
          Polygon([(99, 0), (100, 0), (100, 100), (99, 100)])]

map_4 = [Polygon([(20, 20), (80, 20), (80, 25), (20,25)]),
        Polygon([(20, 60), (80, 70), (50, 80)])]

map_5 = [ Polygon([(30,30),(60,30),(45,60)]) , Polygon([(30,50),(60,50),(45,20)])]

start_point = [10,10,0]
end_point = [90,90,0]

t_start =  time.time()
prm = PRM(num_nodes, distance_radius, space_limits,start_point,end_point, map_3, seed=seed, use_mlp=True)
prm.FindRoadMap('Dijkstra')
t_end =  time.time()

f = prm.plot()

print("SimTime:" ,t_end-t_start)
# prm.plotAsSingles()
date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
plt.title(f"""Elapsed time: {t_end-t_start:.2f} 
            Number of Nodes: {prm.num_nodes}, Connect Radius: {prm.distance_radius}, 
            Theta diff before: {prm.theta_diff_before*(180/np.pi):.1f}, Theta diff after: {prm.theta_diff_after*(180/np.pi):.1f}, 
            Max distance error: {prm.max_dist_error}""")
f.savefig(f"figures/solution_{date}.png", dpi=500)

#plt.show()
