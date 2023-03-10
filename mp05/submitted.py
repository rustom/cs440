# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function

    # first, find the start and waypoint
    start = None
    waypoint = None
    for x in range(maze.size.y):
        for y in range(maze.size.x):
            if maze[x, y] == maze.legend.start:
                start = (x, y)
            if maze[x, y] == maze.legend.waypoint:
                waypoint = (x, y)

    # then, run bfs to find the path from start to waypoint
    queue = []
    queue.append([start])
    visited = set()
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in visited:
            visited.add(node)
            if node == waypoint:
                return path
            for neighbor in maze.neighbors(node[0], node[1]):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return []


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    # first, find the start and waypoint
    start = None
    waypoint = None
    for x in range(maze.size.y):
        for y in range(maze.size.x):
            if maze[x, y] == maze.legend.start:
                start = (x, y)
            if maze[x, y] == maze.legend.waypoint:
                waypoint = (x, y)

    def heuristic(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    # then, run astar to find the path from start to waypoint
    priority_queue = []
    priority_queue.append([start])
    visited = set()
    while priority_queue:
        path = priority_queue.pop(0)
        node = path[-1]
        if node not in visited:
            visited.add(node)
            if node == waypoint:
                return path
            for neighbor in maze.neighbors(node[0], node[1]):
                new_path = list(path)
                new_path.append(neighbor)
                priority_queue.append(new_path)
        priority_queue.sort(key=lambda x: len(x) + heuristic(x[-1][0], x[-1][1], waypoint[0], waypoint[1]))

    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
