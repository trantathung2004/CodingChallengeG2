# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq

import time
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
# print(filenames)

inFile = './input/sample_task2.inp'
# ouFile = './input/sample0.out'
# # ouFile = '/kaggle/working/sample0.out'
# # inFile = '/graph.inp'
# # ouFile = '/graph.out'

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

f = open(inFile, 'r')
line = f.readline()
n, m, k = [int(x) for x in line.split()]
grid = [[None]*m for _ in range(n)]

listBlocks = []

for ii in range(k):
    line = f.readline()
    xx, yy = [int(x) for x in line.split()]
    grid[xx][yy] = float('inf')

lline = f.readline()
# print(lline)
sx, sy, ex, ey = [int(x) for x in lline.split()]

h = int(f.readline())
for _ in range(h):
    line = f.readline()
    x,y,w = [int(x) for x in line.split()]
    grid[x][y] = w

j, k = [int(x) for x in f.readline().split()]

default_weight = 0 if j == 0 else j

for i in range(n):
    for j in range(m):
        if grid[i][j] is None:
            grid[i][j] = default_weight

grid[sx][sy] = -1
grid[ex][ey] = -1
# print_grid(grid)


# # Visualize the grid using matplotlib
# # colors = ['white', 'black', 'red']
# block_color = 'black'
# start_color = 'red'
# default_weight_color = 'white'

# # Create a figure and axis
# fig, ax = plt.subplots()

# max_weight = max(max(cell for cell in row if cell != float('inf')) for row in grid)

# norm = mcolors.Normalize(vmin=0, vmax=max_weight, clip=True)
# mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)  # You can change the colormap to your preference

# # Plot the grid and add black lines
# for i in range(len(grid)):
#     for j in range(len(grid[0])):
#         value = grid[i][j]

#         if value == float('inf'):
#             color = block_color
#         elif value == default_weight:
#             color = default_weight_color
#         elif (i, j) == (sx, sy) or (i, j) == (ex, ey):
#             color = start_color
#         else:
#             color = mapper.to_rgba(value)
        
#         ax.add_patch(plt.Rectangle((j, -i-1), 1, 1, fill=True, color=color))
        
#         # Add black lines to separate squares
    
# for i in range(len(grid)+1):
#         # ax.plot([j, j+1, j+1, j, j], [-i-1, -i-1, -i, -i, -i-1], color='black')
#     ax.plot([0, len(grid[0])], [-i, -i], color='black')
# for j in range(len(grid[0])+1):
#     ax.plot([j, j], [0, -len(grid)], color='black')

# # Set aspect ratio and limits
# ax.set_aspect('equal')
# ax.set_xlim(0, len(grid[0]))
# ax.set_ylim(-len(grid), 0)

# # Hide the axes
# ax.axis('off')

# # Show the plot
# plt.show()



def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# def astar(grid, start, goal):
#     rows, cols = len(grid), len(grid[0])
#     directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Possible movement directions

#     def is_valid(x, y):
#         return 0 <= x < rows and 0 <= y < cols and grid[x][y] != float('inf')  # Check if the position is within bounds and not blocked

#     open_set = [(0, start)]  # Priority queue with initial node
#     came_from = {}  # Dictionary to store parent nodes
#     g_score = {start: 0}  # Cost from start to node

#     while open_set:
#         current_cost, current_node = heapq.heappop(open_set)

#         if current_node == goal:
#             path = []
#             while current_node in came_from:
#                 path.insert(0, current_node)
#                 current_node = came_from[current_node]
#             return path

#         for dx, dy in directions:
#             neighbor = (current_node[0] + dx, current_node[1] + dy)
#             if is_valid(*neighbor):
#                 tentative_g_score = g_score[current_node] + grid[neighbor[0]][neighbor[1]]  # Assuming each move has a cost of 1

#                 if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                     g_score[neighbor] = tentative_g_score
#                     f_score = tentative_g_score + heuristic(neighbor, goal)
#                     heapq.heappush(open_set, (f_score, neighbor))
#                     came_from[neighbor] = current_node

#     return None  # No path found




def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Possible movement directions

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and grid[x][y] != float('inf')  # Check if the position is within bounds and not blocked
    
    def getNeighbours(current):
        n = []
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Possible movement directions

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(*neighbor):
                n.append(neighbor)
        return n
    
    G = {}
    F = {}
    G[start] = 0
    F[start] = heuristic(start, goal)

    closedVertices = set()
    openVertices = set([start])
    cameFrom = {}
    
    while len(openVertices) > 0:
        current = None
        currentFscore = None
        for pos in openVertices:
            if current is None or F[pos] < currentFscore:
                currentFscore = F[pos]
                current = pos
        
        if current == goal:
            path = [current]
            while current in cameFrom:
                current = cameFrom[current]
                path.append(current)
            path.reverse()
            return path, F[goal]  # Done!
        
        openVertices.remove(current)
        closedVertices.add(current)

        for neighbour in getNeighbours(current):
            if neighbour in closedVertices: 
                continue
            candidateG = G[current] + grid[neighbour[0]][neighbour[1]]

            if neighbour not in openVertices:
                openVertices.add(neighbour)
            elif candidateG >= G[neighbour]:
                continue

            cameFrom[neighbour] = current
            G[neighbour] = candidateG
            H = heuristic(neighbour, goal)
            F[neighbour] = G[neighbour] + H
    # raise RuntimeError("A* failed to find a solution")
    return None, None
# Find the shortest path
start = (sx, sy)
goal = (ex, ey)

# Calculate run time
# start_time = time.time()
path, cost = astar(grid, start, goal)
# end_time = time.time()
# runtime = end_time - start_time
# print(f"Runtime: {runtime} seconds")

 
# if path and goal in path:
    # path.remove(goal)

# print shortest path
# if path:
#     print(len(path)+1)
#     # print(str(start[0]) + ' ' + str(start[1]))
#     for node in path:
        
#         print(node[0],node[1])
#         # print()
        
#     print(str(goal[0]) + ' ' + str(goal[1]))
# else:
#     print("No path found.")

with open('sample0.out', 'w') as file:
    if path:
        file.write(str(len(path)) + '\n')
        # file.write(str(start[0])+' ')
        # file.write(str(start[1]) + '\n')
        for node in path:
            for i in range(len(node)):
                file.write(str(node[i]) + ' ')
            file.write('\n')
        # file.write(str(goal[0]) + ' ')
        # file.write(str(goal[1])+'\n')
    else:
        file.write("No path found.\n")
    
# Run A* to find the path
# path = astar(grid, start, goal)

# Mark the path on the grid
# if path and start in path:
#     path.remove(start)
# path_value = -2  # Special marker for the path
# for node in path:
#     x, y = node
#     grid[x][y] = path_value

# Visualization code
# import matplotlib.pyplot as plt

# Define colors
# blocked_color = 'black'
# default_weight_color = 'white'

# path_color = 'yellow'  # Color for the path
# start_color = 'red'
# # end_color = 'blue'

# fig, ax = plt.subplots()

# for i in range(len(grid)):
#     for j in range(len(grid[0])):
#         value = grid[i][j]

#         if value == float('inf'):
#             color = block_color
#         elif value == path_value:
#             color = path_color
#         elif (i, j) == start or (i, j) == goal:
#             color = start_color
#         elif value == default_weight:
#             color = default_weight_color
#         else:
#             color = mapper.to_rgba(value)  # or use a colormap for different weights

#         ax.add_patch(plt.Rectangle((j, -i-1), 1, 1, fill=True, color=color))

# # Add black lines to separate squares
# for i in range(len(grid) + 1):
#     ax.plot([0, len(grid[0])], [-i, -i], color='black')
# for j in range(len(grid[0]) + 1):
#     ax.plot([j, j], [0, -len(grid)], color='black')

# # # Visualize the grid using matplotlib
# # colors = ['white', 'black', 'red', 'yellow']  # Adding yellow for the path

# # # Create a figure and axis
# # fig, ax = plt.subplots()

# # # Plot the grid and add black lines
# # for i in range(len(grid)):
# #     for j in range(len(grid[0])):
# #         square_color = colors[grid[i][j]]
# #         ax.add_patch(plt.Rectangle((j, -i-1), 1, 1, fill=True, color=square_color))

# #         # Add black lines to separate squares
# #         ax.plot([j, j+1, j+1, j, j], [-i-1, -i-1, -i, -i, -i-1], color='black')

# # # Highlight the path in yellow
# # if path:
# #     for node in path:
# #         i, j = node
# #         ax.add_patch(plt.Rectangle((j, -i-1), 1, 1, fill=True, color='yellow'))

# # Set aspect ratio and limits
# ax.set_aspect('equal')
# ax.set_xlim(0, len(grid[0]))
# ax.set_ylim(-len(grid), 0)

# # Add row and column indices to the side
# # for i in range(len(grid)):
# #     ax.text(-0.5, -i - 0.5, str(len(grid) - 1 - i), ha='right', va='center')
# #     ax.text(i + 0.5, 0.5, str(i), ha='center', va='bottom')

# for i in range(len(grid)):
#     ax.text(-0.5, -i - 0.5, str(i), ha='right', va='center')
#     ax.text(i + 0.5, 0.5, str(i), ha='center', va='bottom')
    
# # Hide the axes
# ax.axis('off')

# # Show the plot
# plt.show()

