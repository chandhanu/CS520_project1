from platform import node
from turtle import update
import numpy as np 
from enum import Enum
import random
from collections import deque, namedtuple
import time 
import json 

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


agent2_path = []

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
        try:
            path = [node.state]
        except:
            print(node.state)
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
    
    
    def display(self, m="MAZE"):
        print("---"+m+"---")
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
        if (x.col + 1 < self.cols) and (self.grid[x.row][x.col + 1] != Components.BLOCK) and (self.grid[x.row][x.col + 1] != Components.GHOST):
            coordinates.append(Index(x.row, x.col + 1))
        if (x.row - 1 >= 0) and (self.grid[x.row - 1][x.col] != Components.BLOCK) and (self.grid[x.row - 1][x.col] != Components.GHOST):
            coordinates.append(Index(x.row - 1, x.col))
        if (x.col - 1 >= 0) and (self.grid[x.row][x.col - 1] != Components.BLOCK) and (self.grid[x.row][x.col - 1] != Components.GHOST):
            coordinates.append(Index(x.row, x.col - 1))
        return coordinates
    
    def get_ghost_child(self, x):
        coordinates=[]        
        if (x.row + 1 < self.rows) and (self.grid[x.row + 1][x.col] != Components.GHOST) and (Index(x.row + 1,x.col) != self.goal) :
            if self.grid[x.row + 1][x.col] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row + 1, x.col))
            else:
                coordinates.append(Index(x.row + 1, x.col))
        if  (x.row - 1 >= 0) and (self.grid[x.row - 1][x.col] != Components.GHOST) and (Index(x.row -1 ,x.col) != self.goal) :
            if self.grid[x.row - 1][x.col] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row - 1, x.col))
            else:
                coordinates.append(Index(x.row - 1, x.col))
        if  (x.col + 1 < self.cols) and (self.grid[x.row][x.col + 1] != Components.GHOST) and (Index(x.row ,x.col+1) != self.goal):
            if self.grid[x.row][x.col + 1] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row, x.col + 1))
            else:
                coordinates.append(Index(x.row, x.col + 1))
        if (x.col - 1 >= 0) and (self.grid[x.row][x.col - 1] != Components.GHOST) and (Index(x.row,x.col-1) != self.goal) :
            if self.grid[x.row][x.col - 1] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row, x.col - 1))
            else:    
                coordinates.append(Index(x.row, x.col - 1))

        if len(coordinates) == 0:
            return x
        if len(coordinates) == 1:
            return coordinates[0]
        else:
            if len(coordinates)>0:
                #print("coo--", coordinates)
                c = random.randrange(len(coordinates))
                if c == self.goal and len(coordinates)>1:
                    coordinates.pop(c)
                    c = random.randrange(len(coordinates))
                return coordinates[c]
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

    def dfs_ghost_placement(self, start, destination):

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

    def dfs_agent2(self, start, destination):
        stack = deque()
        stack.append(Node(start, None))
        visited = set([start])
        while stack:
            curr = stack.pop()
            curr_index = curr.state
            #print(curr, self.goal)
            if curr_index == destination:
                return curr
            for child in self.get_children(curr_index):
                #print("curr:", curr, "children:", child.row, child.col)
                if child in visited:
                    continue
                visited.add(child)
                #self.grid[child.row][child.col] = "*"
                stack.append(Node(child, curr))
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
        ghost_lookup_agent2 = {}
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == Components.BLOCK.value:
                    blocked.append((row,col))
                if self.grid[row][col] == Components.EMPTY.value:
                    free.append((row,col))
                    is_path = self.dfs_agent2(Index(row,col), self.goal)
                    if is_path:
                        path = node_to_path(is_path)
                        ghost_lookup_agent2[Index(row, col)] = path
        return free, blocked, ghost_lookup_agent2
                    
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
            #print(a, )
            if not (self.grid[i][j] == Components.GHOST.value or self.grid[i][j] == Components.BLOCK.value or self.grid[i][j] == Components.START.value or self.grid[i][j] == Components.GOAL.value):
                if self.dfs_ghost_placement(Index(i, j), self.start)!=None:
                    self.grid[i][j] = Components.GHOST.value
                    self.ghost_indexes[Index(i,j)] = {"replaced_value": Components.EMPTY }
                    ghost_i+=1
                else:
                    return None
    
    def advance_ghosts(self):
        #print("Advance ghosts")
        ghost_index_list = self.ghost_indexes.keys()
        #print("Ghost indexez-----")
        #print(ghost_index_list)
        #print("------------------")
        delete_ghost_indexes = []
        create_ghost_indexes = {}
        for g_i in ghost_index_list:
            g_v = self.ghost_indexes[g_i]
            #self.display("Advance ghosts")
            ghost_child = self.get_ghost_child(g_i)
            #print("parent:",g_i, "child:",ghost_child)
            #print("Ghost_indexes", g_i, )
            self.grid[g_i.row][g_i.col] = g_v["replaced_value"]
            #self.ghost_indexes[]
            create_ghost_indexes[ghost_child] = {"replaced_value": self.grid[ghost_child.row][ghost_child.col]}
            self.grid[ghost_child.row][ghost_child.col] = Components.GHOST
            delete_ghost_indexes.append(g_i)
        for k,v in create_ghost_indexes.items():
            self.ghost_indexes[k] = v 
        for g_i in delete_ghost_indexes:
            del self.ghost_indexes[g_i]
        '''for k,v in self.ghost_indexes.items():
            print("GHOST i :", k, "Old value :", v)'''

    def shortest_path(self):
        is_path = self.bfs(self.start, self.goal)
        if is_path == None:
            print("Path is blocked")
        else: 
            path = node_to_path(is_path)
            self.mark(path)
            #print(len(path), path)
            self.display()
            self.clear(path)
            print("Path exists")
            return path
    
    def get_next_position(self, x):
        next = self.get_children(x)
        if len(next) :
            return next[0]
        else:
            return None

    '''def agent2_lookup(self, source, dest , lookup):
        curr = source
        path = []
        is_path = self.dfs_agent2(source, dest, lookup)

        pass'''
    
    def agent2(self):
        path = []
        curr = self.start
        while not (curr in self.ghost_indexes) and curr!=self.goal:
        #while self.grid[curr.row][curr.col] != Components.GHOST and curr!=self.goal:
            self.display()
            if self.dfs(curr, self.goal)!=None:
                next_position = self.get_next_position(curr)
                if next_position == None:
                    print("Agent is dead at 1 : ", next_position)
                    print(self.ghost_indexes)
                    return None
                path.append(next_position)
                self.advance_ghosts()
                ################
                if next_position in self.ghost_indexes:
                    print("Agent is dead at 2 : ", next_position)
                    print(self.ghost_indexes)
                    return None
                self.grid[next_position.row][next_position.col] = "A"
                ################
                if next_position == self.goal:
                    return path
                curr = next_position
            else:
                print("CURR", curr, next_position, self.ghost_indexes)
                return None
        if next_position == None:
            print("Agent is dead at : ", next_position)
            print(self.ghost_indexes)
            return None

    def agent2_revamp(self, source): #This waits till the ghost leaves the block (Not in the descriptions)
        path = []
        curr = source 
        global static_i
        global agent2_path
        while curr!=self.goal and (not (curr in self.ghost_indexes )):
            #print("Iteration ", static_i, ":", "curr:",curr)
            static_i+=1
            path = self.dfs_agent2(curr, self.goal)
            if path:
                route = node_to_path(path)
                #print(route)
                for node in route:
                    #self.display()
                    #self.grid[node.row][node.col] = "A"
                    if node in self.ghost_indexes:
                        print("Agent 2 is killed at ", node)
                        self.display("Agent 2 blocked")
                        #next_position = self.get_next_position(curr)
                        #self.agent2_revamp(next_position)
                        #return None 
                    else:
                        agent2_path.append(node)
                    self.advance_ghosts()
                        
                else:
                    return path
            else:
                next_position = self.get_next_position(curr)
                if next_position == None:
                    return None
                else:
                    agent2_path.append(next_position)
                self.advance_ghosts()
                #return self.agent2_revamp(next_position)
    
    def agent1(self):
        path = []
        curr = self.start
        shortest_path = MAZE.shortest_path()
        if shortest_path:
            for node in shortest_path:
                self.display()
                #self.grid[node.row][node.col] = "A"
                if node in self.ghost_indexes:
                    print("Agent 1 died at ", node)
                    return None 
                else:
                    self.advance_ghosts()
            else:
                return shortest_path
        return None
            
if __name__ == "__main__":
    maze = Maze(rows=4, cols=4, probablity_blocked=PROBABLITY_BLOCKED, ghost_count=3)
    # 1. Initiate MAZE 
    maze.initiate()
    FREE_CELLS, BLOCKED_CELLS, ghost_lookup_agent2 =  MAZE.learn_structure()
    MAZE.populate_ghost()
    
    # 3. Display Maze 
    MAZE.display("Initial MAZE")    
    
    # 4. AGENT 1 -- working
    '''path = MAZE.agent1()
    if path:
        print(path)
    else:
        print("Agent 1 Martyred")'''
    #wait = input()
    #print(MAZE.ghost_indexes)
    #path = MAZE.agent2()
    print("Agent2 ")
    #path = MAZE.agent2_lookup(MAZE.start, MAZE.goal, ghost_lookup_agent2)
    #print(path)
    '''path = MAZE.dfs_agent2(MAZE.start, MAZE.goal)
    if path:
        path = node_to_path(path)
        MAZE.mark(path)
        #print(len(path), path)
        MAZE.display()
        MAZE.clear(path)'''

    path = MAZE.agent2()
    #path = MAZE.agent2_revamp(MAZE.start)
    if path:
        path = node_to_path(path)
        MAZE.display("Actual maze")
        MAZE.mark(path)
        #print(len(path), path)
        MAZE.display()
        MAZE.clear(path)
        print(path)
        print("agent2", agent2_path)
        #print(len(path), path)
        MAZE.clear(path)

    else:
        print("Agent 2 Martyred")
    exit(-1)
    '''for i in range(10):
        print(i)
        MAZE.advance_ghosts()
        time.sleep(0.5)'''
    exit(-1)
        

    #print(MAZE.agent1())
    


    #print(FREE_CELLS)
    #print(BLOCKED_CELLS)
    #print(maze)
