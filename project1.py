import numpy as np 
from enum import Enum
import random
#random.seed(100) #- Helps debugging gives same random number everytime 
import sys
from collections import deque, namedtuple
import time 
import json 
from queue import PriorityQueue
from heapq import heappush, heappop
import sys
import operator
sys.setrecursionlimit(1000000)
from multiprocessing import Process

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
agent3_path = []
agent4_path = []
agent5_path = []
agent2_simulation_number = 10

static_i = 0
count_advance_ghost = 5

#All matrix elements are represented as Index variables 
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
        #Displays Maze
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
        
    def get_adjacent_indices(self, i, j, m,n):
        adjacent_indices = []
        if i > 0:
            adjacent_indices.append(Index(i-1,j))
        if i+1 < m:
            adjacent_indices.append(Index(i+1,j))
        if j > 0:
            adjacent_indices.append(Index(i,j-1))
        if j+1 < n:
            adjacent_indices.append(Index(i,j+1))
        
        temp = adjacent_indices[:]

        for x in temp:
            if (self.grid[x.row][x.col] == Components.GHOST):
                adjacent_indices.remove(x)
            elif (self.grid[x.row][x.col] == Components.START) or (self.grid[x.row][x.col] == Components.GOAL):
                adjacent_indices.remove(x)
            elif (self.grid[x.row][x.col] == Components.BLOCK):
                if not random.choice([True, False]):
                    adjacent_indices.remove(x)
        if len(adjacent_indices)==0:
            return Index(i,j)    
        #for node in adjacent
        return random.choice(adjacent_indices)

    def get_ghost_child(self, x):
        coordinates=[]   
        #print(x)     
        if (x.row + 1 < self.rows) and (self.grid[x.row + 1][x.col] != Components.GHOST) and (Index(x.row + 1,x.col) != self.goal) :
            if self.grid[x.row + 1][x.col] == Components.BLOCK:
                if random.choice([True, False]):
                    #print("down no block")
                    coordinates.append(Index(x.row + 1, x.col))
                else:   
                    print("same down")
                    coordinates.append(Index(x.row, x.col))
            else:
                print("down no block")
                coordinates.append(Index(x.row + 1, x.col))
        if  (x.row - 1 >= 0) and (self.grid[x.row - 1][x.col] != Components.GHOST) and (Index(x.row -1 ,x.col) != self.goal) :
            if self.grid[x.row - 1][x.col] == Components.BLOCK:
                if random.choice([True, False]):
                    coordinates.append(Index(x.row - 1, x.col))
                else:    
                    print("same up")
                    coordinates.append(Index(x.row, x.col))
            else:
                print("up no block")
                coordinates.append(Index(x.row - 1, x.col))
        if  (x.col + 1 < self.cols) and (self.grid[x.row][x.col + 1] != Components.GHOST) and (Index(x.row ,x.col+1) != self.goal):
            if self.grid[x.row][x.col + 1] == Components.BLOCK:
                if random.choice([True, False]):
                    coordinates.append(Index(x.row, x.col + 1))
                else:    
                    print("same right")
                    coordinates.append(Index(x.row, x.col))
            else:
                print("right no block")
                coordinates.append(Index(x.row, x.col + 1))
        if (x.col - 1 >= 0) and (self.grid[x.row][x.col - 1] != Components.GHOST) and (Index(x.row,x.col-1) != self.goal) :
            if self.grid[x.row][x.col - 1] == Components.BLOCK:
                if random.choice([True, False]):
                    coordinates.append(Index(x.row, x.col - 1))
                else:    
                    print("same left")
                    coordinates.append(Index(x.row, x.col))
            else:    
                print("left no block")
                coordinates.append(Index(x.row, x.col - 1))
        if len(coordinates) == 0:
            '''print("----yesss-----", x)
            self.grid[x.row][x.col] = "E"
            self.display()
            input()'''
            print("---- idu thaaan prechana")
            return x
        if len(coordinates) == 1:
            print(coordinates)
            input()
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
        while (ghost_i<self.ghost_count):
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
    
    def advance_ghosts(self):
        
        ghost_index_list = self.ghost_indexes.keys()
        delete_ghost_indexes = []
        create_ghost_indexes = self.ghost_indexes.copy()
        for g_i in ghost_index_list:
            g_v = self.ghost_indexes[g_i]
            ghost_child = self.get_adjacent_indices(g_i.row, g_i.col, self.rows, self.cols)
            if ghost_child in self.ghost_indexes:
                continue
            else:
                self.grid[g_i.row][g_i.col] = g_v["replaced_value"]
                create_ghost_indexes[ghost_child] = {"replaced_value": self.grid[ghost_child.row][ghost_child.col]}
                self.grid[ghost_child.row][ghost_child.col] = Components.GHOST
                delete_ghost_indexes.append(g_i)
        for k,v in create_ghost_indexes.items():
            self.ghost_indexes[k] = v 
        for g_i in delete_ghost_indexes:    
            del self.ghost_indexes[g_i]
        
    def shortest_path(self):
        is_path = self.bfs(self.start, self.goal)
        if is_path == None:
            print("Path is blocked")
        else: 
            path = node_to_path(is_path)
            return path
        
    def manhattan_distance(self, goal):
        def distance(ml):
            xdist = abs(ml.col - goal.col)
            ydist = abs(ml.row - goal.row)
            return (xdist + ydist)
        return distance

    def dist(self, ml, goal):
        xdist = abs(ml.col - goal.col)
        ydist = abs(ml.row - goal.row)
        return (xdist + ydist)

    def astar(self, initial, heuristic):
        # priority_queue is where we've yet to go
        priority_queue = PriorityQueue()
        n = Node(initial, None, 0.0, self.dist(initial, self.goal))
        priority_queue.push(Node(initial, None, 0.0, self.dist(initial, self.goal)))
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
                    priority_queue.push(Node(child, current_node, new_cost, self.dist(child, self.goal)))
        return None  # went through everything and never found goal
    
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

    def agent2(self, source):
        curr = source
        global agent2_path
        global static_i 
        count=0
        path = self.astar(curr, self.dist(curr, self.goal))
        #path = self.dfs_agent2(curr, self.goal)
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent2_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks:
                        print("ghosts blocking at ", ghost_blocks)
                        curr = node
                        #input()
                        return self.agent2(curr) # agent2 replans 
                    else:
                        #self.grid[node.row][node.col]="2"
                        agent2_path.append(node)
                    self.advance_ghosts()
            else:
                #BLOCKED
                try:
                    rows = [1, -1, 0, 0]
                    cols = [0, 0, 1, -1]
                    closest_ghost_distance_all_path = float('-infinity')
                    for i,j in zip(rows, cols):
                        curr_row = curr.row+i
                        curr_col = curr.col+j
                        new_index = Index(curr_row,curr_col)
                        new_path = self.astar(new_index, self.dist(new_index, self.goal))
                        if  new_path:
                            ghost_blocks_in_path = common_member(node_to_path(new_path), self.ghost_indexes.keys())
                            closest_ghost_distance = float("infinity")
                            for id, ghost in enumerate(ghost_blocks_in_path):
                                closest_ghost_distance = min(closest_ghost_distance, self.dist(new_index, ghost))
                            if closest_ghost_distance >= closest_ghost_distance_all_path:
                                closest_ghost_distance_all_path = closest_ghost_distance
                                curr = new_index
                                path = new_path
                    #closest_ghost_distance
                except:
                    #self.advance_ghosts() 
                    count +=1
                    if count ==50:
                        return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

    def agent2_simulation(self, source, return_path, count = 0 ):
        # Agent 2 with cut short search spaces 
        curr = source
        global agent2_path
        global static_i 
        
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            #path = self.dfs_agent2(curr, self.goal)
            path = self.astar(curr, self.manhattan_distance(self.goal))
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    if node == self.goal:
                        return return_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks and count == 5:
                        #print("ghosts blocking at ", ghost_blocks)
                        curr = node
                        #count+=1
                        return self.agent2_simulation(curr,return_path, count+1) # agent2 replans 
                    else:
                        return_path.append(node)
                    #self.advance_ghosts() # keeping maze constant 
            else:
                #BLOCKED
                #self.advance_ghosts() # keeping maze constant - Cutting short the search space 
                count +=1
                if count ==5:
                    return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

    def get_survivability(self, children, simulation_number, method = "astar"):
        s_index = {} # Survivability_index
        global agent2_path
        agent2_path = []
        #print("------S - Index --------")
        for child in children: 
            for i in range(simulation_number):
                if method == "astar":
                    path = self.astar(child, self.manhattan_distance(self.goal))
                else:
                    path = self.agent2_simulation(child,[])
                if path :
                    if child in s_index:
                        s_index[child] +=1
                    else:
                        s_index[child] = 1
                    
                    #print(child, "has path")
                else:
                    #print("no solution agent 3 sucks at ", child )
                    self.advance_ghosts()
                    print(".", end="")
                    #self.display()
        return s_index     

    def get_next_child(self, s_index_max):
        next_child = next(iter(s_index_max))
        max_child_value = s_index_max[next_child]
        l = []
        for k,v in  s_index_max.items():
            if v == max_child_value:
                l.append(k)
        return random.choice(l)

    def agent3_working_but_low_efficiency(self, start, destination, prev):
        curr = start 
        global agent3_path
        global agent2_simulation_number
        global count_advance_ghost
        temp_list = []
        self.grid[prev.row][prev.col]=" "
        #print(curr)
        self.grid[curr.row][curr.col]="+"
        self.display()
        #input()
        if curr == self.goal:
            temp_list.append(curr)
            return temp_list
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            children = self.get_children(curr)
            if children:
                if self.goal in children:
                    temp_list.append(curr)
                    temp_list.append(self.goal)
                    #agent3_path.extend(temp_list)
                    return temp_list
                s_index = self.get_survivability(children, agent2_simulation_number) # Survivability_index 
                s_index_max = dict( sorted(s_index.items(), key=operator.itemgetter(1),reverse=True))
                
                if len(s_index_max):
                    next_child = self.get_next_child(s_index_max)
                    self.advance_ghosts()
                    temp_list = self.agent3_old(next_child, destination,curr)
                    #agent3_path.extend(self.agent3(next_child, destination))
                else:
                    # Blocked path no child
                    if count_advance_ghost!=0:
                        self.advance_ghosts() #Agent 3 waits 
                        count_advance_ghost-=1
                        temp_list = self.agent3_old(curr, destination, prev)
                    else:
                        #self.advance_ghosts()
                        return []
            else:
                print("Blocked at ", curr)
                self.advance_ghosts()
                return []
        else:
            if curr == self.goal:
                return agent3_path.extend(temp_list) if temp_list != None else agent3_path
            print("Dead at ", curr)
            return []
        
        return agent3_path.extend(temp_list) if temp_list != None else agent3_path

        if curr in self.ghost_indexes:
            print("Killed at ", curr)
            #exit(-1)
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

    def agent3(self,start):
        curr = start
        global agent3_path
        global static_i 
        global count_advance_ghost
        count=0
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            #path = self.astar(curr, self.manhattan_distance(self.goal))
            path = self.astar(curr, self.manhattan_distance(self.goal))
            
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent3_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks:
                        #print("ghosts blocking at ", ghost_blocks)
                        ###################
                        
                        if curr == self.goal:
                            temp_list.append(curr)
                            return temp_list
                        if curr!=self.goal and (not (curr in self.ghost_indexes )):
                            children = self.get_children(curr)
                            if children:
                                if self.goal in children:
                                    temp_list.append(curr)
                                    temp_list.append(self.goal)
                                    #agent3_path.extend(temp_list)
                                    return temp_list
                                s_index = self.get_survivability(children, agent2_simulation_number, "agent2_simulated") # Survivability_index 
                                s_index_max = dict( sorted(s_index.items(), key=operator.itemgetter(1),reverse=True))
                                
                                if len(s_index_max):
                                    next_child = self.get_next_child(s_index_max)
                                    self.advance_ghosts()
                                    temp_list = self.agent3(next_child)
                                    #agent3_path.extend(self.agent3(next_child, destination))
                                else:
                                    # Blocked path no child
                                    if count_advance_ghost!=0:
                                        self.advance_ghosts() #Agent 3 waits 
                                        count_advance_ghost-=1
                                        temp_list = self.agent3(curr)
                                    else:
                                        #self.advance_ghosts()
                                        return []
                            else:
                                print("Blocked at ", curr)
                                self.advance_ghosts()
                                return []
                        ###################
                        curr = node
                        #input()
                        return self.agent2(curr) # agent3 replans # break the barrier 
                    else:
                        agent3_path.append(node)
                    self.advance_ghosts()

            else:
                #BLOCKED
                #print("ELSEeeeeeeeeeeeeeeeeee")
                self.advance_ghosts() #Agent 4 waits 
                count +=1
                if count ==100:
                    return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

    def agent4(self,start):
        curr = start
        global agent3_path
        global static_i 
        global count_advance_ghost
        global agent2_simulation_number
        count=0
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            #path = self.astar(curr, self.manhattan_distance(self.goal))
            path = self.astar(curr, self.manhattan_distance(self.goal))
            
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent3_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks:
                        #print("ghosts blocking at ", ghost_blocks)
                        ###################
                        
                        if curr == self.goal:
                            temp_list.append(curr)
                            return temp_list
                        if curr!=self.goal and (not (curr in self.ghost_indexes )):
                            children = self.get_children(curr)
                            if children:
                                if self.goal in children:
                                    temp_list.append(curr)
                                    temp_list.append(self.goal)
                                    #agent3_path.extend(temp_list)
                                    return temp_list
                                s_index = self.get_survivability(children, 2*agent2_simulation_number) # Survivability_index 
                                s_index_max = dict( sorted(s_index.items(), key=operator.itemgetter(1),reverse=True))
                                
                                if len(s_index_max):
                                    next_child = self.get_next_child(s_index_max)
                                    self.advance_ghosts()
                                    temp_list = self.agent4(next_child)
                                    #agent3_path.extend(self.agent3(next_child, destination))
                                else:
                                    # Blocked path no child
                                    if count_advance_ghost!=0:
                                        self.advance_ghosts() #Agent 3 waits 
                                        count_advance_ghost-=1
                                        temp_list = self.agent4(curr)
                                    else:
                                        #self.advance_ghosts()
                                        return []
                            else:
                                print("Blocked at ", curr)
                                self.advance_ghosts()
                                return []
                        ###################
                        curr = node
                        #input()
                        return self.heuristic(curr)  
                    else:
                        agent3_path.append(node)
                    self.advance_ghosts()

            else:
                #BLOCKED
                self.advance_ghosts() #Agent 4 waits 
                count +=1
                if count ==100:
                    return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

    def heuristic(self, start): # Increased threshold for agent 4
        curr = start
        global agent3_path
        global static_i 
        count=0
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            path = self.astar(curr, self.manhattan_distance(self.goal))
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent3_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks:
                        #print("ghosts blocking at ", ghost_blocks)
                        curr = node
                        #input()
                        return self.heuristic(curr) # repeated astar replans 
                    else:
                        agent3_path.append(node)
                    self.advance_ghosts()

            else:
                #BLOCKED
                try: # Attempts to move away from the nearest ghost 
                    rows = [1, -1, 0, 0]
                    cols = [0, 0, 1, -1]
                    closest_ghost_distance_all_path = float('-infinity')
                    for i,j in zip(rows, cols):
                        curr_row = curr.row+i
                        curr_col = curr.col+j
                        new_index = Index(curr_row,curr_col)
                        new_path = self.astar(new_index, self.dist(new_index, self.goal))
                        if  new_path:
                            ghost_blocks_in_path = common_member(node_to_path(new_path), self.ghost_indexes.keys())
                            closest_ghost_distance = float("infinity")
                            for id, ghost in enumerate(ghost_blocks_in_path):
                                closest_ghost_distance = min(closest_ghost_distance, self.dist(new_index, ghost))
                            if closest_ghost_distance >= closest_ghost_distance_all_path:
                                closest_ghost_distance_all_path = closest_ghost_distance
                                curr = new_index
                                path = new_path
                    #closest_ghost_distance
                except: # if no solution, it will wait for the ghost to move 
                    self.advance_ghosts() 
                    count +=1
                    if count ==200: # agent 4 can wait a longer time than agent 3
                        return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

    def agent5(self, source):
        # this is against agent 2, except the blocked ghosts are not known 
        curr = source
        global agent5_path
        global static_i 
        count=0
        while curr!=self.goal and (not (curr in self.ghost_indexes )):
            path = self.astar(curr, self.dist(curr, self.goal))
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent5_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    #print(ghost_blocks)
                    #input()
                    if ghost_blocks: #Ignore the ghost in block and dont replan 
                        if not (self.ghost_indexes[list(ghost_blocks)[-1]]["replaced_value"] == Components.BLOCK):
                            print("ghosts blocking at ", ghost_blocks)
                            curr = node
                        #input()
                        return self.agent5(curr) # agent2 replans 
                    else:
                        #self.grid[node.row][node.col]="2"
                        agent5_path.append(node)
                    self.advance_ghosts()
            else:
                #BLOCKED
                self.advance_ghosts() 
                count +=1
                if count ==50:
                    return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

    def agent5_againt_low_info(self,start):
        # this is against agent 5, except the blocked ghosts are not known 
        curr = start
        global agent3_path
        global static_i 
        global count_advance_ghost
        count=0
        if curr!=self.goal and (not (curr in self.ghost_indexes )):
            #path = self.astar(curr, self.manhattan_distance(self.goal))
            path = self.astar(curr, self.manhattan_distance(self.goal))
            
            static_i+=1
            if path:
                count=0
                route = node_to_path(path)
                for node in route : 
                    #print(node, self.goal)
                    if node == self.goal:
                        return agent3_path
                    #finding ghost in path 
                    ghost_blocks = common_member(route, self.ghost_indexes.keys())
                    if ghost_blocks:
                        #print("ghosts blocking at ", ghost_blocks)
                        ###################
                        if  (self.ghost_indexes[list(ghost_blocks)[-1]]["replaced_value"] == Components.BLOCK):
                            continue
                        if curr == self.goal:
                            temp_list.append(curr)
                            return temp_list
                        if curr!=self.goal and (not (curr in self.ghost_indexes )):
                            children = self.get_children(curr)
                            if children:
                                if self.goal in children:
                                    temp_list.append(curr)
                                    temp_list.append(self.goal)
                                    #agent3_path.extend(temp_list)
                                    return temp_list
                                s_index = self.get_survivability(children, agent2_simulation_number, "agent2_simulated") # Survivability_index 
                                s_index_max = dict( sorted(s_index.items(), key=operator.itemgetter(1),reverse=True))
                                
                                if len(s_index_max):
                                    next_child = self.get_next_child(s_index_max)
                                    self.advance_ghosts()
                                    temp_list = self.agent3(next_child)
                                    #agent3_path.extend(self.agent3(next_child, destination))
                                else:
                                    # Blocked path no child
                                    if count_advance_ghost!=0:
                                        self.advance_ghosts() #Agent 3 waits 
                                        count_advance_ghost-=1
                                        temp_list = self.agent3(curr)
                                    else:
                                        #self.advance_ghosts()
                                        return []
                            else:
                                print("Blocked at ", curr)
                                self.advance_ghosts()
                                return []
                        ###################
                        curr = node
                        #input()
                        return self.agent2(curr) # agent3 replans # break the barrier 
                    else:
                        agent3_path.append(node)
                    self.advance_ghosts()

            else:
                #BLOCKED
                #print("ELSEeeeeeeeeeeeeeeeeee")
                self.advance_ghosts() #Agent 4 waits 
                count +=1
                if count ==100:
                    return None
        if curr in self.ghost_indexes:
            print("Killed at ", curr)
        return None

if __name__ == "__main__":
    # python3 project1
    switch = {
        "1": "agent1",
        "2": "agent2",
        "3": "agent3",
        "4": "agent4",
        "5": "agent5",
        "6": "agent5_2"
    }
    if len(sys.argv) ==1 :
        print("---------------------------------------")
        print("Usage: python3 project1.py 1|2|3|4|5|6")
        print("\t1 - Agent1")
        print("\t2 - Agent2")
        print("\t3 - Agent3")
        print("\t4 - Agent4")
        print("\t5 - Agent5")
        print("---------------------------------------")
        sys.exit(-1)
    choice = switch[sys.argv[1]]
    maze = Maze(rows=51, cols=51, probablity_blocked=PROBABLITY_BLOCKED, ghost_count=10)
    # 1. Initiate MAZE 
    maze.initiate()
    FREE_CELLS, BLOCKED_CELLS =  MAZE.learn_structure()
    
    # 2. populate ghost
    MAZE.populate_ghost()
    #print(len(MAZE.ghost_indexes))
    
    # 3. Display Maze 
    print("MAZE Configurations")
    print("Total cell # is", MAZE.rows* MAZE.cols)
    MAZE.display("VALID MAZE")   
    #input() 
    #input()
    #exit(-1)
    if choice == "agent1":
        # 4. AGENT 1 -- working
        is_path = MAZE.agent1()
        if is_path:
            d = {}
            for i in is_path:
                d[i]=MAZE.grid[i.row][i.col]
                MAZE.grid[i.row][i.col] = "1"
            MAZE.display("AGENT1 - path")
            for i in d.keys():
                MAZE.grid[i.row][i.col] = d[i]
        else:
            print("Agent 1 Martyred")

    elif choice == "agent2":
        # 5.Agent2 
        print("Agent2 ")
        path = MAZE.agent2(MAZE.start) #working 
        if path:
            d = {}
            for i in path:
                d[i]=MAZE.grid[i.row][i.col]
                MAZE.grid[i.row][i.col] = "2"
            MAZE.display("AGENT2 - path")
            for i in d.keys():
                MAZE.grid[i.row][i.col] = d[i]

        else:
            MAZE.display("Agent 2 death")
            
    elif choice == "agent3":
        # 6. Agent 3
        path = MAZE.agent3(MAZE.start)
        if path:
            d = {}
            for i in path:
                d[i]=MAZE.grid[i.row][i.col]
                MAZE.grid[i.row][i.col] = "3"
            MAZE.display("AGENT3 - path")
            for i in d.keys():
                MAZE.grid[i.row][i.col] = d[i]
        else:
            MAZE.display("Agent 3 death")

    elif choice == "agent4":
        # 7. Agent 3        
        path = MAZE.agent4(MAZE.start)
        if path:
            d = {}
            for i in path:
                d[i]=MAZE.grid[i.row][i.col]
                MAZE.grid[i.row][i.col] = "4"
            MAZE.display("AGENT4 - path")
            for i in d.keys():
                MAZE.grid[i.row][i.col] = d[i]
        else:
            MAZE.display("Agent 4 death")

    elif choice == "agent5":
        # 8. Agent2 - Agent 5 
        print("Agent5 ")
        path = MAZE.agent5(MAZE.start) #working 
        if path:
            d = {}
            for i in path:
                d[i]=MAZE.grid[i.row][i.col]
                MAZE.grid[i.row][i.col] = "5"
            MAZE.display("AGENT5 - path")
            for i in d.keys():
                MAZE.grid[i.row][i.col] = d[i]

        else:
            MAZE.display("Agent 5 death")

    elif choice == "agent5_2":
        # 8. Agent2 - Agent 5 
        print("Agent5 ")
        path = MAZE.agent5(MAZE.start) #working 
        if path:
            d = {}
            for i in path:
                d[i]=MAZE.grid[i.row][i.col]
                MAZE.grid[i.row][i.col] = "6"
            MAZE.display("AGENT5 - path")
            for i in d.keys():
                MAZE.grid[i.row][i.col] = d[i]

        else:
            MAZE.display("Agent 5 death")

    sys.exit()
