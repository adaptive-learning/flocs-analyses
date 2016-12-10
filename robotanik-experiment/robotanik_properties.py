from colorconsole import terminal
from time import sleep

def simulate(problem, solution, animate=False, max_steps = 200):
    """
    simulates given solution of the problem, animation works only when run from command line
    """
    #print("simulating", solution)
    board = get_board(problem)
    #get number of flowers
    flowersLeft = problem["board"].count("R")
    flowersLeft +=problem["board"].count("G")
    flowersLeft +=problem["board"].count("B")

    #create structure for simulation info
    simulationInfo = {"maxRecDepth":0,"Fsteps":0,"LRsteps":0,"functionExecuted":[True,False,False,False,False],"functionOrdering":[1,9,9,9,9]}
    simulationInfo["commandsExecuted"] = list(map(lambda x: [False for y in x],solution))
    simulationInfo["conditionWasFalse"] = list(map(lambda x: [False for y in x],solution))

    #current recursion depth
    recDepth = 0

    #ordering of functions - number which will be associated  with the next function run
    #F1 must alway be F1, so we start from 2
    order = 2

    col = problem["robotCol"]
    row = problem["robotRow"]
    rot = problem["robotDir"]
    simulationInfo["visited"] = [(row,col)] #visited fields

    steps = list(reversed(solution[0])) #stack of steps to simulate
    stack = [[1,0]] #stack of functions and positions of simulated steps
    if animate:
            outputBoard(board,row,col,rot,True)
    #while there are flowers and steps to make, make a step
    step_counter = 0
    while (steps!=[] and flowersLeft>0 and step_counter < max_steps):
        step_counter += 1
        if animate:
            outputBoard(board,row,col,rot)
            sleep(0.01)
        condition, step = steps[-1] #find the current condition and command
        steps.pop() #remove command from steps
        actf,actp = stack[-1] #find the function and position of simulated step
        #if recursion is ending, pop from stack
        if step in "vt": stack.pop()
        #otherwise move to next position in the same function
        else: stack[-1][1]+=1

        #if the color of current field is the same as the condition or there is no condition
        if condition==board[row][col] or condition=="_":
            if step!="v" and step!="t":
                #if it is the actual commad (not the info about recurssion ending) set the command to executed
                #command is not considered executed if it is recoloring to the same color
                if step!= board[row][col]: simulationInfo["commandsExecuted"][actf-1][actp] = True
            if step.isdigit():
                #if the command is F1 - F5

                if steps!=[] and steps[-1][1]!="v" and steps[-1][1]!="t":
                    #if its not the tail recursion
                    recDepth+=1 #increase recursion depth
                    #update maximum recursion depth
                    simulationInfo["maxRecDepth"] = max(recDepth,simulationInfo["maxRecDepth"])
                    steps.append(("_","v")) #append information about recursion ending
                else: steps.append(("_","t")) #otherwise it is tail rec - append info about tail rec ending

                steps+= list(reversed(solution[int(step)-1])) #add commands from called function on steps stack
                stack.append([int(step),0]) #add first position of function on the top of positions stack
                simulationInfo["functionExecuted"][int(step)-1] = True #function was executed rigth now

                #if function was not executed before, set its execution order
                if simulationInfo["functionOrdering"][int(step)-1]==9:
                    simulationInfo["functionOrdering"][int(step)-1]=order
                    order+=1
            else:
                #command is NOT a function call
                if step=="v":
                    #recursion ended - decrease depth
                    recDepth -= 1
                if step=="L":
                    #rotate left and increase LR steps
                    rot=(rot-1)%4
                    simulationInfo["LRsteps"]+=1
                if step=="R":
                    #rotate right and increase LR steps
                    rot=(rot+1)%4
                    simulationInfo["LRsteps"]+=1
                if step=="F":
                    #move in the direction given by rotation
                    col+=[1,0,-1,0][rot]
                    row+=[0,1,0,-1][rot]
                    #RADEK hack
                    if col < 0 or row < 0 or col >= 16 or row >= 11:
                        break
                    #TOM hack: stop when the Robotanik crashes into a wall
                    if board[row][col] == ' ':
                        break
                    simulationInfo["Fsteps"]+=1

                    #add new position to visited
                    if simulationInfo["visited"][-1]!=(row,col):
                        simulationInfo["visited"].append((row,col))

                if step in "rgb":
                    #recolor if necessary
                    if board[row][col] != step:
                        board[row] = board[row][0:col]+step+board[row][col+1:]
                if board[row][col].istitle():
                    #if there was a flower, it is now taken
                    board[row] = board[row][0:col]+board[row][col].lower()+board[row][col+1:]
                    flowersLeft-=1
        else:
            #command was not executed because the condition was False
            #therefore this condition was useful
            simulationInfo["conditionWasFalse"][actf-1][actp] = True
    if animate:
            outputBoard(board,row,col,rot)
    simulationInfo["stackLeft"] = recDepth #save recursion depth at the end of simulation
    simulationInfo["visitedSet"] = set(simulationInfo["visited"]) #compute the set of visited positions
    return simulationInfo

def outputBoard(board,row,col,rot,complete=False):
    """
    ASCII draws the current board to the console with robotanist position specified by row, col, rot
    if complete then whole boards is redrawn, otherwise only the fields next to robot (to avoid blinking)
    """
    screen = terminal.get_terminal()
    if complete:
        screen.gotoXY(0,0)
        screen.clear()
    for i in range(len(board)):
        for j in range(len(board[0])):
            if not complete:
                if abs(i-row)>1 or abs(j-col)>1: continue
            c = board[i][j]
            screen.set_color(terminal.colors["WHITE"], terminal.colors['BLACK'])
            if c.lower()=="r":
                screen.set_color(terminal.colors["WHITE"], terminal.colors['RED'])
            if c.lower()=="g":
                screen.set_color(terminal.colors["WHITE"], terminal.colors['GREEN'])
            if c.lower()=="b":
                screen.set_color(terminal.colors["WHITE"], terminal.colors['BROWN'])
            if (row,col) == (i,j):
                if rot==0: screen.print_at(2*j,i, "->")
                if rot==1: screen.print_at(2*j,i, "\\/")
                if rot==2: screen.print_at(2*j,i, "<-")
                if rot==3: screen.print_at(2*j,i, "/\\")
            else:
                if c.istitle(): screen.print_at(2*j,i, "{}") #there is a flower
                else: screen.print_at(2*j,i, "  ")
    screen.set_color(terminal.colors["WHITE"], terminal.colors['BLACK'])


def get_board(problem):
    return parse_board(problem['board'])

def parse_board(board_string):
    return [board_string[i:i+16] for i in range(0,16*12,16)]
