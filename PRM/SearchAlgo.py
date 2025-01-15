import heapq

def Dijkstra(nodes, edges, start_node_index, end_node_index):
    distances = {node_index: float('inf') for node_index in range(len(nodes))}
    distances[start_node_index] = 0
    
    priority_queue = [(0, start_node_index)]
    previous = {node_index: None for node_index in range(len(nodes))}
    edges_by_node = {node_index: None for node_index in range(len(nodes))}
    
    # Iterate until the priority queue is empty
    while priority_queue:
        current_distance, current_node_index = heapq.heappop(priority_queue)
        
        if current_node_index == end_node_index:
            break
        
        for edge in edges:
            if (edge[0] != current_node_index):
                continue
            
            neighbor_index = edge[1]
            distance_to_neighbor = current_distance + edge[4]
            
            # If this distance is shorter than the previously known shortest distance to the neighbor, update it
            if distance_to_neighbor < distances[neighbor_index]:
                distances[neighbor_index] = distance_to_neighbor
                previous[neighbor_index] = current_node_index
                edges_by_node[neighbor_index] = edge
                heapq.heappush(priority_queue, (distance_to_neighbor, neighbor_index))
    
    # Reconstruct the shortest path
    shortest_path = []
    current_node_index = end_node_index
    while current_node_index is not None and edges_by_node[current_node_index] is not None:
        if (current_node_index == start_node_index):
            break
        shortest_path.append(edges_by_node[current_node_index])
        current_node_index = previous[current_node_index]
    shortest_path.reverse()
    
    return shortest_path, distances[end_node_index]
