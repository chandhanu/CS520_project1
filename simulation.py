import project1 
import project1_bhuvan
import json
from multiprocessing import Process
import logging 
import sys
sys.setrecursionlimit(1000000)
agent1 = []
agent2 = []
agent3 = []
agent4 = []

static_c =0 
fc=0
def generate_test_values():
    maze_size = [] 
    blocks_ghost_maze_dict =   {}
    for maze_size in range(51,52):
        total_cells = maze_size*maze_size
        free_cells = int(total_cells*0.25)
        gc = [i for i in range(0, free_cells)]
        #gc = [i for i in range(1, int((free_cells**(0.5))))]
        blocks_ghost_maze_dict[maze_size] = {"ghost_count":gc}
    return blocks_ghost_maze_dict

def run_simultation(switch="agent3"):
    simulation_values = generate_test_values()
    print(simulation_values)
    #w = input()
    global fc
    for maze_size, values in simulation_values.items():
        values["data"] = {}
        #for gc in [8]:
        for gc in (values["ghost_count"]):
        #for gc in ([25, 50, 100, 125, 150, 200]):
            print("Size:",maze_size,"ghost:",gc)
            success_count = int(0)
            failure_count = 0
            verdict = None
            data={
                "success_rate":0, 
                "success_count":0
                }
            values["data"][gc]=data

            #logger.info("Total cell # is", maze.rows* maze.cols)
            for x in range(10):
                maze = project1_bhuvan.Maze(rows=maze_size, cols=maze_size, probablity_blocked=0.28, ghost_count=gc)
                # 1. Initiate MAZE 
                maze.initiate()
                FREE_CELLS, BLOCKED_CELLS =  maze.learn_structure()
                maze.populate_ghost()

                # 2. Display Maze 
                #print("----MAZE----")
                #print("Total cell # is", maze.rows* maze.cols)
                         
                #maze.display("VALID MAZE")

                # 3. Agent 1
                if switch=="agent1":
                    path = maze.agent1()
                    if path:
                        values["data"][gc]["success_count"]+=1
                        verdict = True
                    # 4. Agent 2
                elif switch == "agent2":
                    path = maze.age2(maze.start)
                    if path:
                        values["data"][gc]["success_count"]+=1
                        verdict = True

                # 5. Agent 3
                elif switch == "agent3":
                    #path = maze.agent3(maze.start, maze.goal)
                    path = maze.agent3(maze.start)
                    if path:
                        values["data"][gc]["success_count"]+=1
                        verdict = True

                # 6. Agent 4
                elif switch == "agent4":
                    path = maze.agent4(maze.start)
                    if path:
                        values["data"][gc]["success_count"]+=1
                        verdict = True

            success_rate = values["data"][gc]["success_count"]/10
            values["data"][gc]["success_rate"] = success_rate
            with open("New2"+switch+".txt", "a") as myfile:
                myfile.write("\n")
                myfile.write("GC: "+str(gc)+" Success Rate : "+str(success_rate))
            if success_rate <=0:
                fc+=1
                if fc>3:
                    with open("New2"+switch+".json", "w") as outfile:
                        json.dump(simulation_values, outfile)
                    exit(1)
            else:
                fc = 0
            print("***********************************")
            print("Ghost", gc,"success", success_rate)
            #logger.info("Ghost", gc,"success", success_rate)
            print("***********************************")
            #input()


            
if __name__ == "__main__":

    processes = list()
    for i in range(0, 1):
        process = Process(target=run_simultation,args=(sys.argv[1],))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    

    #for size, data in simulation_values[50].items():

                
