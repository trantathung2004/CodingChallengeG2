import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

def color_grid(N, M, grid):
    # Function to get the available colors for a cell
    def get_available_colors(x, y):
        colors = set(range(N * M))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] >= 0:
                if grid[nx][ny] in colors:
                    colors.remove(grid[nx][ny])
        return colors

    # Color the grid
    for x in range(N):
        for y in range(M):
            if grid[x][y] == 0:  # If cell is not blocked
                available_colors = get_available_colors(x, y)
                grid[x][y] = min(available_colors)  # Assign the lowest available color

    return grid

listInp = []
for dirname, _, filenames in os.walk('./sample/'):
    for filename in filenames:
        if filename[-3:]  == 'inp':
            listInp.append(os.path.join(dirname, filename))
listInp = sorted(listInp)
for idx, filename in enumerate(listInp):
    input_file = filename
    counter = filename.split('.')[1].split('/')[-1]
    counter = int(counter[6:])

    f = open(filename, 'r')
    line = f.readline()
    n, m, k = [int(x) for x in line.split()]
    # listBlocks = []
    grid = [[0] * m for _ in range(n)]
    blocks = []
    for ii in range(k):
        line = f.readline()
        xx, yy = [int(x) for x in line.split()]
        grid[xx][yy] = -1
        blocks.append((xx, yy))

    lline = f.readline()       

    colored_grid = color_grid(n, m, grid)

    if not os.path.exists('./result'):
            os.mkdir('./result')

    filename = f'./result/sample{counter}.out'  

    with open(filename, 'w') as file:
        for row in colored_grid:
            file.write(" ".join(map(str, row)) + '\n')