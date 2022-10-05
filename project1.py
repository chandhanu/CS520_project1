import numpy as np 
from enum import Enum
import random
from collections import deque, namedtuple
import time 

PROBABLITY_BLOCKED = 0.28
PROBABLITY_UNBLOCKED = 1-PROBABLITY_BLOCKED
PROBABLITY_GHOST_ENTERING_BLOCK = 0.5

MAZE = None
BLOCKED_CELLS = []
FREE_CELLS = []
LEFT = (-1, 0)
RIGHT = (1,0)
UP = (0,1)
DOWN = (-1,0)



static_i = 0

Index = namedtuple('Index', ['row', 'col'])

class Components(str, Enum):
    START = "S"
    GOAL = "G"
    BLOCK = "#"
    PATH = "+"
    EMPTY = "."
    GHOST = "X"


class Node:
    def __init__(self, state, parent ):
        self.state = state
        self.parent =  parent

def node_to_path(node):
        path = [node.state]
        # work backwards from end to front
        while node.parent is not None:
            node = node.parent
            path.append(node.state)
        path.reverse()
        return path

class Maze:
    def __init__(self, rows, cols, probablity_blocked, ghost_count):
        self.rows = rows
        self.cols = cols
        self.P = probablity_blocked
        self.start = Index(0,0)
        self.goal = Index(rows-1,cols-1)
        self.grid = [[Components.EMPTY for c in range(cols)] for r in range(rows)]
        self.ghost_count = ghost_count
        self.ghost_indexes = {}
        self.ghost_entering_block_probablity = 0.5
    
    
    def display(self):
        print("---MAZE---")
        for i in self.grid:
            print(" ".join(i))
    
    def generate_maze(self):
        return np.random.choice([Components.BLOCK.value, Components.EMPTY.value], 
            size =(self.row, self.col), 
            p=[PROBABLITY_BLOCKED, PROBABLITY_UNBLOCKED])
    
    def fill_blocks(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if random.uniform(0, 1.0) < self.P:
                    self.grid[row][col] = Components.BLOCK
                    '''BLOCKED_CELLS.append((row,col))
                else:
                    FREE_CELLS.append((row,col))'''
    
    def get_children(self, x):
        coordinates=[]
        if (x.row + 1 < self.rows) and (self.grid[x.row + 1][x.col] != Components.BLOCK) and (self.grid[x.row + 1][x.col] != Components.GHOST):
            coordinates.append(Index(x.row + 1, x.col))
        if (x.row - 1 >= 0) and (self.grid[x.row - 1][x.col] != Components.BLOCK) and (self.grid[x.row - 1][x.col] != Components.GHOST):
            coordinates.append(Index(x.row - 1, x.col))
        if (x.col + 1 < self.cols) and (self.grid[x.row][x.col + 1] != Components.BLOCK) and (self.grid[x.row][x.col + 1] != Components.GHOST):
            coordinates.append(Index(x.row, x.col + 1))
        if (x.col - 1 >= 0) and (self.grid[x.row][x.col - 1] != Components.BLOCK) and (self.grid[x.row][x.col - 1] != Components.GHOST):
            coordinates.append(Index(x.row, x.col - 1))
        return coordinates
    
    def get_ghost_children(self, x):
        coordinates=[]
        if (x.row + 1 < self.rows) and (self.grid[x.row + 1][x.col] != Components.GHOST):
            if self.grid[x.row + 1][x.col] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row + 1, x.col))
            else:
                coordinates.append(Index(x.row + 1, x.col))
        if (x.row - 1 >= 0) and (self.grid[x.row - 1][x.col] != Components.GHOST):
            if self.grid[x.row - 1][x.col] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row - 1, x.col))
            else:
                coordinates.append(Index(x.row - 1, x.col))
        if (x.col + 1 < self.cols) and (self.grid[x.row][x.col + 1] != Components.GHOST):
            if self.grid[x.row][x.col + 1] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row, x.col + 1))
            else:
                coordinates.append(Index(x.row, x.col + 1))
        if (x.col - 1 >= 0) and (self.grid[x.row][x.col - 1] != Components.GHOST):
            if self.grid[x.row][x.col - 1] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row, x.col - 1))
            else:    
                coordinates.append(Index(x.row, x.col - 1))
        return coordinates

    def mark(self, path):
        for maze_location in path:
            self.grid[maze_location.row][maze_location.col] = Components.PATH
        self.grid[self.start.row][self.start.col] = Components.START
        self.grid[self.goal.row][self.goal.col] = Components.GOAL

    def clear(self, path):
        for maze_location in path:
            self.grid[maze_location.row][maze_location.col] = Components.EMPTY
        self.grid[self.start.row][self.start.col] = Components.START
        self.grid[self.goal.row][self.goal.col] = Components.GOAL

    def dfs(self, start, destination):

        stack = deque()
        stack.append(start)
        visited = set([start])
        while stack:
            curr = stack.pop()
            #print(curr, self.goal)
            if curr == destination:
                return curr
            for child in self.get_children(curr):
                #print("curr:", curr, "children:", child.row, child.col)
                if child in visited:
                    continue
                visited.add(child)
                #self.grid[child.row][child.col] = "*"
                stack.append(child)
        return None 

    def bfs(self, start, destination):
        queue = deque()
        queue.append(Node(start, None))
        visited = set([start])
        while queue:
            curr = queue.popleft()
            curr_index = curr.state
            #print(curr, self.goal)
            if curr_index == destination:
                #print("YESSSS")
                return curr
            for child in self.get_children(curr_index):
                #print("curr:", curr, "children:", child.row, child.col)
                if child in visited:
                    continue
                visited.add(child)
                #self.grid[child.row][child.col] = "*"
                #self.display()
                #time.sleep(0.1)
                queue.append(Node(child, curr))
        return None 
    
    def is_valid_maze_temp(self, start_position=(0,0)):
        visited = set([start_position])
        fringe = [start_position]
        while fringe:
            i,j = fringe.pop(0)
            visited.add((i,j))
            for di,dj in [RIGHT, DOWN, LEFT, UP]:
                ni,nj = i+di, j+dj
                if (ni,nj) in visited:
                    continue
                if ni<0 or nj<0 or ni>=self.rows or nj>=self.cols:
                    continue
                if self.grid[ni][nj] == Components.GOAL.value:
                    return True
                if self.grid[ni][nj] == Components.BLOCK.value:
                    continue
                if self.grid[ni][nj] == Components.EMPTY.value:
                    visited.add((ni,nj))
                    #self.grid[i][j] = "*"
                    fringe.append((ni,nj))
        return False
    
    def is_valid_maze(self):
        return self.dfs(self.start, self.goal)

    def learn_structure(self):
        free, blocked = [], []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == Components.BLOCK.value:
                    blocked.append((row,col))
                if self.grid[row][col] == Components.EMPTY.value:
                    free.append((row,col))
        return free, blocked 
                    
    def initiate(self):
        #maze = self.generate_maze()
        self.fill_blocks()
        self.grid[self.start.row][self.start.col] = Components.START
        self.grid[self.goal.row][self.goal.col] = Components.GOAL
        #self.display()
        # 1 - Check the validity of the maze 
        validity = self.is_valid_maze()
        global static_i
        #print("Generate Maze Attempt:", static_i, " -- Validation Failed No Path Found")
        if validity is None: 
            static_i+=1
            self.grid = [[Components.EMPTY for c in range(self.cols)] for r in range(self.rows)]
            self.initiate()
        else:
            global MAZE
            MAZE = self
        return

    def populate_ghost(self):
        ghost_i = 0
        while (ghost_i<self.ghost_count):
            a = random.choice(FREE_CELLS)
            i, j = a
            if not (self.grid[i][j] == Components.GHOST.value or self.grid[i][j] == Components.BLOCK.value):
                if self.dfs(Index(i, j), self.start)!=None:
                    self.grid[i][j] = Components.GHOST.value
                    self.ghost_indexes[Index(i,j)] = {"replaced_value": Components.EMPTY, }
                    ghost_i+=1
    
    def advance_ghost(self):
        for g_i, g_v in self.ghost_indexes.items():
            print(g_i, type(g_i))


    def run_agent1(self):
        is_path = self.bfs(self.start, self.goal)
        if is_path == None:
            print("Agent1 martyred")
        else: 
            path = node_to_path(is_path)
            self.mark(path)
            #print(len(path), path)
            self.display()
            self.clear(path)
            print("Agent 00 7 survived")

if __name__ == "__main__":
    maze = Maze(rows=10, cols=10, probablity_blocked=PROBABLITY_BLOCKED, ghost_count=5)
    # 1. Initiate MAZE 
    maze.initiate()
    FREE_CELLS, BLOCKED_CELLS =  MAZE.learn_structure()
    #print("START", MAZE.start)
    #print("GOAL", MAZE.goal)
    print(MAZE.start.row, "---------")
    # 2. Populate Ghost 
    MAZE.populate_ghost()
    
    # 3. Display Maze 
    MAZE.display()    

    # 4. Agent 1 
    MAZE.run_agent1()
    print(MAZE.ghost_indexes)
    


    #print(FREE_CELLS)
    #print(BLOCKED_CELLS)
    #print(maze)
