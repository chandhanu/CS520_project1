from platform import node
from turtle import update
import numpy as np 
from enum import Enum
import random
import sys
from collections import deque, namedtuple
import time 
import json 
from queue import PriorityQueue
from heapq import heappush, heappop
import sys
sys.setrecursionlimit(1000000)

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
    EMPTY = " "
    GHOST = "X"


class Node:
    def __init__(self, state, parent, cost = 0.0, heuristic = 0.0  ):
        self.state = state
        self.parent =  parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        return (a_set & b_set)
    else:
        return None

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

class PriorityQueue():
    def __init__(self):
        self.container = []

    @property
    def empty(self):
        return not self.container  # not is true for empty container

    def push(self, item):
        heappush(self.container, item)  # in by priority

    def pop(self):
        return heappop(self.container)  # out by priority

    def __repr__(self):
        return repr(self.container)

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
        self.free_cells = self.blocked_cells = 0
    
    
    def display(self, m="MAZE"):
        print("---"+m+"---")
        c=0
        
        for i in self.grid:
            print(" ".join(i))
            c+=1
    
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
    def get_children_revamp(self, x):
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
                    coordinates.append(Index(x.row, x.col))
            else:
                coordinates.append(Index(x.row + 1, x.col))
        if  (x.row - 1 >= 0) and (self.grid[x.row - 1][x.col] != Components.GHOST) and (Index(x.row -1 ,x.col) != self.goal) :
            if self.grid[x.row - 1][x.col] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row - 1, x.col))
                else:    
                    coordinates.append(Index(x.row, x.col))
            else:
                coordinates.append(Index(x.row - 1, x.col))
        if  (x.col + 1 < self.cols) and (self.grid[x.row][x.col + 1] != Components.GHOST) and (Index(x.row ,x.col+1) != self.goal):
            if self.grid[x.row][x.col + 1] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row, x.col + 1))
                else:    
                    coordinates.append(Index(x.row, x.col))
            else:
                coordinates.append(Index(x.row, x.col + 1))
        if (x.col - 1 >= 0) and (self.grid[x.row][x.col - 1] != Components.GHOST) and (Index(x.row,x.col-1) != self.goal) :
            if self.grid[x.row][x.col - 1] == Components.BLOCK:
                if random.uniform(0, 1.0) < PROBABLITY_GHOST_ENTERING_BLOCK:
                    coordinates.append(Index(x.row, x.col - 1))
                else:    
                    coordinates.append(Index(x.row, x.col))
            else:    
                coordinates.append(Index(x.row, x.col - 1))
        if len(coordinates) == 0:
            '''print("----yesss-----", x)
            self.grid[x.row][x.col] = "E"
            self.display()
            input()'''
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
    def dfs_agent2_revamp(self, start, destination):
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
                    '''is_path = self.dfs_agent2(Index(row,col), self.goal)
                    if is_path:
                        path = node_to_path(is_path)
                        ghost_lookup_agent2[Index(row, col)] = path'''
        self.free_cells = free
        self.blocked_cells = blocked
        return free, blocked#, ghost_lookup_agent2
                    
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
        d={}
        #print(len(self.free_cells))
        while (ghost_i<=self.ghost_count):
            a = random.choice(self.free_cells)
            i, j = a
            #print(a, ghost_i)
            if not (self.grid[i][j] == Components.GHOST.value or self.grid[i][j] == Components.BLOCK.value or self.grid[i][j] == Components.START.value or self.grid[i][j] == Components.GOAL.value):
                #ghost_to_start_path = self.dfs_ghost_placement(Index(i, j), self.start)
                self.grid[i][j] = Components.GHOST.value
                self.ghost_indexes[Index(i,j)] = {"replaced_value": Components.EMPTY }
                ghost_i+=1
                self.free_cells.remove((i,j))
                continue
                if self.dfs_ghost_placement(Index(i, j), self.start)!=None:
                    self.grid[i][j] = Components.GHOST.value
                    self.ghost_indexes[Index(i,j)] = {"replaced_value": Components.EMPTY }
                    ghost_i+=1
                    self.free_cells.remove((i,j))
                else:
                    if ghost_i in d:
                        d[ghost_i] +=1
                        if d[ghost_i]>50: #Threshold quantity to place the ghost with valid path from ghost to goal
                            return
                    else:
                        d[ghost_i]=1
        #print(len(self.free_cells))
    
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
            #self.mark(path)
            #print(len(path), path)
            #self.display()
            #self.clear(path)
            #print("Path exists")
            return path
    
    def get_next_position(self, x):
        next = self.get_children(x)
        if len(next) :
            return next[0]
        else:
            return None
    
    def manhattan_distance(self, goal):
        def distance(ml):

            xdist = abs(ml.col - goal.col)
            ydist = abs(ml.row - goal.row)
            return (xdist + ydist)
        return distance

    def astar(self, initial, heuristic):
        # priority_queue is where we've yet to go
        priority_queue = PriorityQueue()
        n = Node(initial, None, 0.0, heuristic(initial))
        priority_queue.push(Node(initial, None, 0.0, heuristic(initial)))
        # explored is where we've been
        explored = {initial: 0.0}

        # keep going while there is more to explore
        while not priority_queue.empty:
            current_node = priority_queue.pop()
            current_state = current_node.state
            # if we found the goal, we're done
            if current_state == self.goal:
                return current_node
            # check where we can go next and haven't explored
            for child in self.get_children(current_state):
                new_cost: float = current_node.cost + 1  # 1 assumes a grid, need a cost function for more sophisticated apps

                if child not in explored or explored[child] > new_cost:
                    explored[child] = new_cost
                    priority_queue.push(Node(child, current_node, new_cost, heuristic(child)))
        return None  # went through everything and never found goal
    
    def agent2(self):
        path = []
        curr = self.start         
        global agent2_path
        while curr !=self.goal and  (not (curr in self.ghost_indexes )):
            path = self.dfs_agent2(curr, self.goal)
            if path:
                route = node_to_path(path)
                w = input()
                #print(route)
                for i in route:
                    if i in self.ghost_indexes:
                        print("AGENT2 is dead", i)
                        self.display("AGENT2 DEAD")
                        return None
                    else:
                        agent2_path.append(i)
                    self.advance_ghosts()
                else:
                    return path
            else:
                print("AGENT2 blocked", curr)
                #self.display("AGENT2 DEAD")
                print("GHOST\n-------", self.ghost_indexes)
                #self.find_nearest_ghost()
                return None
        return None
    
    def get_nearest_ghost(self, curr):
        min_x = 99999
        return_path = []
        for k,v in self.ghost_indexes.items():
            if v["replaced_value"] == Components.EMPTY:
                path = self.bfs(curr, k )
                if path:
                    route = node_to_path(path)
                    if len(route)<min_x:
                        min_x=len(route)
                        return_path = route
                        return_key = k
        if k :
            return k
        else:
            return None

    
    def age2(self, source):
        curr = source
        global agent2_path
        global static_i 
        count=0
        while curr!=self.goal and (not (curr in self.ghost_indexes )):
            print("node------------------------------------------------------", curr)
            path = self.dfs_agent2(curr, self.goal)
            #print("globallllllll-",static_i)
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for i in route:
                    self.grid[i.row][i.col]="+"
                for i in self.ghost_indexes.keys():
                    self.grid[i.row][i.col]="X"
                #self.display("PlannedRoot")
                
                for i in self.ghost_indexes.keys():
                    self.grid[i.row][i.col]=self.ghost_indexes[i]["replaced_value"]
                for i in route:
                    self.grid[i.row][i.col]=" "

                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent2_path
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks:
                        print("ghosts blocking at ", ghost_blocks)
                        curr = node
                        #input()
                        return self.age2(curr)
                    else:
                        #self.grid[node.row][node.col]="2"
                        agent2_path.append(node)
                    self.advance_ghosts()
                    self.display()
                    #input()
                '''for i in route[1:]:
                    if i == self.goal:
                        return agent2_path
                    self.grid[i.row][i.col]="2"
                    agent2_path.append(i)
                    self.advance_ghosts()
                    self.display()
                input()'''
            else:
                #BLOCKED
                print("ELSEeeeeeeeeeeeeeeeeee")
                self.advance_ghosts() #Agent 2 waits 
                count +=1
                if count ==200:
                    return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None



    def agent2_revamp(self, source): #This waits till the ghost leaves the block (Not in the descriptions)
        path = []
        curr = source 
        global agent2_path
        while curr!=self.goal and (not (curr in self.ghost_indexes )):
            #print("Iteration ", static_i, ":", "curr:",curr)

            path = self.dfs_agent2(curr, self.goal)
            if path:
                route = node_to_path(path)
                #print(route)
                for node in route:
                    #self.display()
                    #self.grid[node.row][node.col] = "A"
                    if node in self.ghost_indexes:
                        print("Agent 2 is killed at ", node)
                        #self.display("Agent 2 blocked")
                        #next_position = self.get_next_position(curr)
                        #self.agent2_revamp(next_position)
                        return None 
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
                #self.display()
                #self.grid[node.row][node.col] = "1"
                if node in self.ghost_indexes:
                    print("Agent 1 died at ", node)
                    #self.display("A1 Death")
                    #print(len(self.ghost_indexes),self.ghost_indexes)
                    return None 
                
                self.advance_ghosts()
            else:
                return shortest_path
        return None

            
if __name__ == "__main__":
    
    maze = Maze(rows=51, cols=51, probablity_blocked=PROBABLITY_BLOCKED, ghost_count=300)
    # 1. Initiate MAZE 
    maze.initiate()
    FREE_CELLS, BLOCKED_CELLS =  MAZE.learn_structure()
    MAZE.populate_ghost()
    print(len(MAZE.ghost_indexes))
    exit(-1)
    # 3. Display Maze 
    print("MAZE Configurations")
    print("Total cell # is", MAZE.rows* MAZE.cols)
    print("BLOCKED cells # is",len(BLOCKED_CELLS))
    print("FREE cells # is",len(FREE_CELLS))
    MAZE.display("VALID MAZE")    
    
    #exit(-1)
    # 4. AGENT 1 -- working
    is_path = MAZE.agent1()
    if is_path:
        d = {}
        for i in is_path:
            d[i]=MAZE.grid[i.row][i.col]
            MAZE.grid[i.row][i.col] = "+"
        MAZE.display("AGENT1 - path")
        for i in d.keys():
            MAZE.grid[i.row][i.col] = d[i]
    else:
        print("Agent 1 Martyred")
    # 5.Agent2 
    print("Agent2 ")
    path = MAZE.age2(MAZE.start) #working 
    if path:
        d = {}
        for i in path:
            d[i]=MAZE.grid[i.row][i.col]
            MAZE.grid[i.row][i.col] = "+"
        MAZE.display("AGENT2 - path")
        for i in d.keys():
            MAZE.grid[i.row][i.col] = d[i]

    else:
        MAZE.display("A2 death")
        print("Agent 2 Martyred")
    
    exit(-1)
    
    #exit(-1)
    # A start check 
    print("Astar")
    distance = MAZE.manhattan_distance(MAZE.goal)
    #print("Manhattan distance ", distance(MAZE.goal))
    path = MAZE.astar(MAZE.start, distance)
    print(path.state)
    if path:
        path2 = node_to_path(path)
        print(path2)
        #MAZE.display("Actual maze")
        MAZE.mark(path2)
        #print(len(path), path)
        MAZE.display()
        MAZE.clear(path2)
    else:
        print("Astar failed")
    '''for i in range(10):
        print(i)
        MAZE.advance_ghosts()
        time.sleep(0.5)'''
        

    #print(MAZE.agent1())
    


    #print(FREE_CELLS)
    #print(BLOCKED_CELLS)
    #print(maze)
