# CodingChallengeG2
## This is done by: **Tran Tat Hung** and **Luu Nguyen Chi Duc**

(**Note that the detailed description for each task can be found as ReadMe.docx in the folder of each task**)

### Task 1: Finding Path in Simple Graphs
Unravel the mysteries of simple graphs and enhance your skills in finding the most efficient paths. From basic algorithms to advanced techniques, this session is designed to sharpen your graph traversal skills. Assuming from a cell, you could move to 4 adjacency cells (up, down, left, right). Your mission is to find a path from point A to point B.

### Task 2: Finding Path in Complex Graphs
Ready for a more challenging adventure? Session 2 delves into the complexities of navigating. Explore algorithms that tackle the challenges posed by complex graphs and uncover strategies to find the optimal paths. Sharpen your problem-solving skills as you navigate through real-world scenarios. In this session, you are not just finding a path from point A to point B, your task is also finding a path that takes the least steps and the least power.

### Task 3: Coloring Vertex and Optimization
Colors bring graphs to life! In Session 3, we explore the world of graph coloring and optimization. Learn techniques to color vertices efficiently and discover how proper coloring can lead to optimized solutions.

### Problem Definition:
**Given:**

*   A 2D grid representing the environment where the robot operates.
*   The grid contains cells, and each cell can either be empty or occupied by an obstacle.
*   The robot starts at a specified initial cell.
*   The goal is to reach a target cell in the grid.

**Objective:**

*   Find the optimal path for the robot to navigate from the starting cell to the target cell while avoiding obstacles and lowest cost.

**Graph Representation:**

Nodes (Vertices):
*   Each cell in the grid is represented as a node in the graph.
  
*   Empty cells are valid nodes, while cells with obstacles are excluded from the graph.

Edges:
*   Edges connect neighboring cells in the grid.
*   Possible movements (edges) include going up, down, left, or right, depending on the grid structure.
*   Diagonal movements can also be considered based on the allowed navigation rules.

Graph Properties:
*   The graph is undirected since movements are bidirectional between adjacent cells.
*   Each edge has a weight representing the cost or distance of moving from one cell to another. The weight might be uniform (e.g., all edges have the same weight) or variable based on factors such as terrain difficulty.

Start and Target:
*   The initial cell serves as the starting node in the graph.
*   The target cell is the destination or goal node.
*   In a grid-based environment, a robot needs to navigate from a start point to a goal point while avoiding obstacles. We could formulate this problem as a graph problem.
