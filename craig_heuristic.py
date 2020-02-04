from queue import PriorityQueue
import numpy as np
from itertools import product

unique_val = []

class State(object):
    def __init__(self, value, parent, start = 0, goal = 0):
        self.children = []
        self.parent = parent
        self.value = value
        self.dist = 0
        if parent:
            self.path = parent.path[:]
            self.path.append(value)
            self.start = parent.start
            self.goal = parent.goal
        else:
            self.path = [value]
            self.start = start
            self.goal = goal
    
    def GetDist(self, board):
        dist_list = []  # empty list. will hold distances between 2 points on flow-free board
        # find unique values on board (ex: 1, 2, 3)
        # ignores 0, which symbolizes empty spaces
        for sublist in board:
            for val in sublist:
                if val not in unique_val:
                    unique_val.append(val)
            unique_val.remove(0)
        # for unique_values, finds coordinates for each pair of unique values
        for i in unique_val:
            coord = np.where(board == i)
            coord = np.asarray(coord).T.tolist()
            print(coord, 'COORD')
            # counts number of vertical steps (rows)
            x_val = abs(coord[0][0] - coord[1][0])
            # counts number of horizontal steps (columns)
            y_val = abs(coord[0][1] - coord[1][1])
            # calculates total distance
            distance = y_val + x_val
            # appends distances to list
            dist_list.append(distance)
        return dist_list
        # starter = dist_list.index(min(dist_list)) + 1
        # print(starter)

    def isSafe(self, board, point):
        #The board being the matrix
        #The point being the point chosen from the path(s)
        print('Chosen point to move to is: ', point)
        x = point[0]
        y = point[1]
        print(board[x][y])
        if board[x][y] != 0:
            return False
        else:
            return True


    def eval_function(self, d_list, board):
        # pulls index for the minimum distance. For our board, it will be an index of 2. Symbolizes that
        # the pair of three's on the board are closest together
        print(d_list) 
        start = np.argmin(d_list)
        # replace the value at index 2 with 100, so that next time we search for the minimum in the list, it won't pick the
        # value at index 2 again
        d_list[start] = 100
        # the d_list is now [3, 4, 100]. Therefore the next minimum is the value 3 at index zero. Index zero corresponds
        # with the pair of ones on the board.
        print(d_list)
        print(start)
        start = start + 1
        print(start)
        return(start)

    def neighbors(self, index):
        print('The point is at: ',index)
        N = len(index)
        for relative_index in product((-1, 0, 1), repeat=N):
            if not all(i == 0 for i in relative_index):
                everyone = list(i + i_rel for i, i_rel in zip(index, relative_index))
                yield everyone

    def isSafe(self, board, point):
        print('Chosen point to be moved to: ', point)
        x = point[0]
        y = point[1]
        print(board[x][y])
        if board[x][y] != 0:
            return False
        else:
            return True

    def maxBoardLength(self, board):
        max_rows = len(board)
        max_columns = max(map(len, board))
        return max_rows, max_columns

    def removeCheats(self, start, neighborList, point):
        updated_list = []
        max_rows, max_columns = self.maxBoardLength(start)
        for i in range(len(neighborList)):
            #Removes the diagonals
            if abs(sum(neighborList[i]) - sum(point)) != 2 and abs(sum(neighborList[i]) - sum(point)) !=0:  #Removes the negatives
                if sum(neighborList[i]) > 0:
                    #Takes care of the out of grid
                    if neighborList[i][0] < max_rows and neighborList[i][1] < max_columns:
                        updated_list.append(neighborList[i])
        #print(updated_list)
        return updated_list


    def CreateChildren(self):
        pass


class HillSolver:
    def __init__(self, start, goal):
        self.path          = []
        self.visitedQueue  = []
        self.priorityQueue = PriorityQueue()
        self.start         = start
        self.goal          = goal

    def solve(self):
        initialState = State(self.start, 0, self.start, self.goal) #0 because not a parent
        coordinates = initialState.GetDist(self.start)
        intial_mover = initialState.eval_function(coordinates, self.start)
        neighbor_list = list(initialState.neighbors((1,1)))
        moves = initialState.removeCheats(self.start, neighbor_list, (1,1))
        print('Available moves are: ', moves)
        
        #Makes a point be chosen. For testing purposes I made it choose (2,1)
        point_chosen = moves[3]
        if initialState.isSafe(self.start, moves[3]) == True:
            print('Valid')
        else:
            print('Not valid')
        

if __name__ == '__main__':
    HillStart =  np.array([[1, 2, 0, 0],
                   [0, 3, 0, 0],
                   [0, 1, 3, 0],
                   [2, 0, 0, 0]])
    
    HillGoal = np.array([[1,2,2,2],
                        [1,3,3,2],
                        [1,1,3,2],
                        [2,2,2,2]])
    
    test = HillSolver(HillStart, HillGoal)
    test.solve()
