import numpy as np 
from enum import Enum
import random
from collections import deque, namedtuple

PROBABLITY_BLOCKED = 0.28
PROBABLITY_UNBLOCKED = 1-PROBABLITY_BLOCKED

MAZE = None

LEFT = (-1, 0)
RIGHT = (1,0)
UP = (0,1)
DOWN = (-1,0)

class Components(str, Enum):
    START = "S"
    GOAL = "G"
    BLOCK = "#"
    PATH = "1"
    EMPTY = "."

static_i = 0

'''class Index():
    def __init__(self, i, j):
        self.row = i
        self.col = j'''

Index = namedtuple('Index', ['row', 'col'])

class Node:
    def __init__(self, ):
        pass

class Maze:
    def __init__(self, rows, cols, probablity_blocked):
        self.rows = rows
        self.cols = cols
        self.P = probablity_blocked
        self.start = Index(0,0)
        self.goal = Index(rows-1,cols-1)
        self.grid = [[Components.EMPTY for c in range(cols)] for r in range(rows)]
    
    
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
    
    def get_children(self, x):
        coordinates=[]
        if x.row + 1 < self.rows and self.grid[x.row + 1][x.col] != Components.BLOCK:
            coordinates.append(Index(x.row + 1, x.col))
        if x.row - 1 >= 0 and self.grid[x.row - 1][x.col] != Components.BLOCK:
            coordinates.append(Index(x.row - 1, x.col))
        if x.col + 1 < self.cols and self.grid[x.row][x.col + 1] != Components.BLOCK:
            coordinates.append(Index(x.row, x.col + 1))
        if x.col - 1 >= 0 and self.grid[x.row][x.col - 1] != Components.BLOCK:
            coordinates.append(Index(x.row, x.col - 1))
        return coordinates
    
    def dfs(self):
        stack = deque()
        stack.append(self.start)
        visited = set([self.start])
        while stack:
            curr = stack.pop()
            #print(curr, self.goal)
            if curr == self.goal:
                return curr
            for child in self.get_children(curr):
                #print("curr:", curr, "children:", child.row, child.col)
                if child in visited:
                    continue
                visited.add(child)
                #self.grid[child.row][child.col] = "*"
                stack.append(child)
        return None 

    def bfs(self):
        queue = deque()
        queue.append(self.start)
        visited = set([self.start])
        while queue:
            curr = queue.popleft()
            #print(curr, self.goal)
            if curr == self.goal:
                print("YESSSS")
                return curr
            for child in self.get_children(curr):
                #print("curr:", curr, "children:", child.row, child.col)
                if child in visited:
                    continue
                visited.add(child)
                self.grid[child.row][child.col] = "*"
                queue.append(child)
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
        return self.dfs()
                    
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



if __name__ == "__main__":
    #print(Components.BLOCK.value)
    maze = Maze(rows=10, cols=10, probablity_blocked=PROBABLITY_BLOCKED)
    maze.initiate()
    #maze.display()
    # ACTUAL --- 
    print("FINAL MAZE")
    MAZE.display()    
    #print(maze)
