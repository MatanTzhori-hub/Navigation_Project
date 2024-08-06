from PRM.Solver_Minimize import *
from PRM.PRM import *
from shapely.geometry import LineString, Polygon, Point

num_nodes = 1200
distance_radius = 10  # Adjust distance radius as needed
space_limits = [(0, 100), (0, 100)]  # Limits for x and y coordinates

# Obstacles
map_1 = [ Polygon([(70.0, 50.0), (60.0, 65.0), (40.0, 65.0), (30.0, 50.0), (40.0, 35.0), (60.0, 35.0)]) ]

map_2 = [ Polygon([(40, 40), (60, 60), (40, 60), (60,40)]) ]

map_3 = [ Polygon([(0, 30), (70, 30), (70, 35), (0, 35)]),
          Polygon([(100, 65), (30, 65), (30, 70), (100, 70)])]

map_4 = [Polygon([(20, 20), (80, 20), (80, 25), (20,25)]),
        Polygon([(20, 60), (80, 70), (50, 80)])]

map_5 = [ Polygon([(30,30),(60,30),(45,60)]) , Polygon([(30,50),(60,50),(45,20)])]

prm = PRM(num_nodes, distance_radius, space_limits, map_5)
prm.FindRoadMap([10,20,0], [90,90,0], 'Dijkstra')
# prm.FindRoadMap([90,5,0], [10,95,0], 'A_Star')
prm.plot()
plt.show()