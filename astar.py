import heapq

def heuristic(start, goal):
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])

def find_path(start, goal, snake_body, grid_width, grid_height):
    def get_neighbors(node):
        x, y = node
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]  # Right, Left, Down, Up
        
        # Filter out neighbors that are outside the grid boundaries or are part of the snake's body
        neighbors = [
            (nx, ny) for nx, ny in neighbors 
            if (0 <= nx < grid_width) and (0 <= ny < grid_height) 
                and ((nx, ny) not in snake_body)
        ]
        
        return neighbors

    open_list = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node != start:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current_node):
            new_cost = cost_so_far[current_node] + 1  # Assuming all edges have unit weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current_node

    return None  # No path found
