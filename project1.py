import numpy as np 
from enum import Enum
import random

PROBABLITY_BLOCKED = 0.28
PROBABLITY_UNBLOCKED = 1-PROBABLITY_BLOCKED
class Components(str, Enum):
    START = "S"
    GOAL = "G"
    BLOCK = "#"
    PATH = "1"
    EMPTY = "0"

class Index:
    def __init__(self, i, j):
        self.i = i 
        self.j = j

class Maze:
    def __init__(self, rows, cols, probablity_blocked):
        self.rows = rows
        self.cols = cols
        self.P = probablity_blocked
        self.start = Index(0,0)
        self.goal = Index(rows-1,cols-1)
        self.grid = [[Components.EMPTY for c in range(cols)] for r in range(rows)]
        #self.fill_blocks()
    
    
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
            for column in range(self.cols):
                if random.uniform(0, 1.0) < self.P:
                    self.grid[row][column] = Components.BLOCK
    
    def is_valid(self, start_position=(0,0)):
        seen = set([start_position])
        queue = [start_position]
        while queue:
            i,j = queue.pop(0)
            seen.add((i,j))
            for di,dj in [(1,0),(0,-1),(-1,0),(0,1)]:
                ni,nj = i+di, j+dj
                if (ni,nj) in seen:
                    continue
                if ni<0 or nj<0 or ni>=self.rows or nj>=self.cols:
                    continue
                if self.grid[ni][nj] == "G":
                    return True
                if self.grid[ni][nj] == "#":
                    continue
                if self.grid[ni][nj] == "0":
                    seen.add((ni,nj))
                    queue.append((ni,nj))
        return False
    
    def initiate(self):
        #maze = self.generate_maze()
        self.display()
        self.fill_blocks()
        self.grid[self.start.i][self.start.j] = Components.START
        self.grid[self.goal.i][self.goal.j] = Components.GOAL
        validity = self.is_valid()
        print(validity)


if __name__ == "__main__":
    print(Components.BLOCK.value)
    maze = Maze(rows=10, cols=10, probablity_blocked=PROBABLITY_BLOCKED)
    maze.initiate()
    maze.display()
    
    #print(maze)
