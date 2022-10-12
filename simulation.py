import project1 
import json

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
        gc = [i for i in range(0, free_cells,5)]
        #gc = [i for i in range(1, int((free_cells**(0.5))))]
        blocks_ghost_maze_dict[maze_size] = {"ghost_count":gc}
    return blocks_ghost_maze_dict

            
if __name__ == "__main__":
    simulation_values = generate_test_values()
    print(simulation_values)
    w = input()
    for maze_size, values in simulation_values.items():
        values["data"] = {}
        #for gc in [8]:
        for gc in (values["ghost_count"]):
            print("Size:",maze_size,"ghost:",gc)
            success_count = int(0)
            failure_count = 0
            verdict = None
            data={
                "success_rate":0, 
                "failure_rate":0,
                "success_count":0,
                "failure_count":0
                }
            values["data"][gc]=data
            
            for x in range(100):
                maze = project1.Maze(rows=maze_size, cols=maze_size, probablity_blocked=0.28, ghost_count=gc)
                # 1. Initiate MAZE 
                maze.initiate()
                FREE_CELLS, BLOCKED_CELLS =  maze.learn_structure()
                #print(FREE_CELLS)
                maze.populate_ghost()

                # 2. Display Maze 
                print("----MAZE----")
                print("Total cell # is", maze.rows* maze.cols)
                 
                static_c +=1
                if static_c==2000:
                    break
                         
                #maze.display("VALID MAZE")

                # 3. Agent 1
                path = maze.agent1()
                if path:
                    values["data"][gc]["success_count"]+=1
                    verdict = True
                else:
                    values["data"][gc]["failure_count"]+=1
                    verdict = False
                # 4. Agent 2
                '''path = maze.age2(maze.start)
                if path:
                    values["data"][gc]["success_count"]+=1
                    verdict = True
                else:
                    values["data"][gc]["failure_count"]+=1
                    verdict = False'''
            print(success_count)
            success_rate = values["data"][gc]["success_count"]/100
            values["data"][gc]["success_rate"] = success_rate

            values["data"][gc]["failure_rate"] = values["data"][gc]["failure_count"]/100
            if success_rate <=0:
                fc+=1
                if fc>3:
                    with open("agent1.json", "w") as outfile:
                        json.dump(simulation_values, outfile)
                    exit(1)
            else:
                fc = 0
            print("***********************************")
            print("Ghost", gc,"success", success_rate)
            print("***********************************")
            #input()

        
    with open("agent1.json", "w") as outfile:
        json.dump(simulation_values, outfile)

    #for size, data in simulation_values[50].items():

                
