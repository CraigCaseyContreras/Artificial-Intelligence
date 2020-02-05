import numpy as np
from itertools import product

# Our initial flow free state
matrix = np.array([[1, 2, 0, 0],
                   [0, 3, 0, 0],
                   [0, 1, 3, 0],
                   [2, 0, 0, 0]])

unique_val = []  # empty list. will hold unique values from matrix

#ADJACENTS = {(1,0), (0,1), (-1,0), (0,-1)}

#function that returns the neighbors
def neighbors(index):
    print('The point is at: ',index)
    N = len(index)
    for relative_index in product((-1, 0, 1), repeat=N):
        if not all(i == 0 for i in relative_index):
            everyone = list(i + i_rel for i, i_rel in zip(index, relative_index))
            yield everyone

# function makes sure we don't step off the board
# I haven't actually used this anywhere....
# Maybe do this AFTER the AI has chosen a point to go to. And this function determines if the point is safe or not.

def maxBoardLength(board):
    max_rows = len(board)
    max_columns = max(map(len, board))
    return max_rows, max_columns

def isSafe(board, point):
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

def cost_function(board):
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


def eval_function(d_list, board):
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


def removeCheats(neighborList, point):
    updated_list = []
    max_rows, max_columns = maxBoardLength(matrix)
    for i in range(len(neighborList)):
        #Removes the diagonals
        if abs(sum(neighborList[i]) - sum(point)) != 2 and abs(sum(neighborList[i]) - sum(point)) !=0:
            #Removes the negatives
            if sum(neighborList[i]) > 0:
                #Takes care of the out of grid
                if neighborList[i][0] < max_rows and neighborList[i][1] < max_columns:
                    updated_list.append(neighborList[i])
    #print(updated_list)
    return updated_list

#Returns the list of distances NOT the elements or parents
#What this did is essentially pick the shortest distance. So since the list was (3,4,2). This 2 corresponds to the index 2. At index 2, we have the element '3'.
#So it chose 3. Thus, 3 is our 'first mover'. Remeber that '3' has a distnce of 2 to finish. We go by indices.

def main():
    initial_mover = eval_function(cost_function(matrix), matrix) 
    print(initial_mover)

    #We picked the one with the least amount of children. Get its coordinates. Then, we have to get all of its neighbors. So since we know it is at (1,1)
    #Need to make function that finds out what the coordinate of the one with least children is
    #We established that 3 should move first. Now we just have to figure out which 3? 
    neighbor_list = list(neighbors((1,1)))
    print('The neighbors are: ',neighbor_list)
    #Now to remove illegal diagonal elements
    neighbor_list = removeCheats(neighbor_list, (1,1))
    print('The neighbors are: ',neighbor_list)
    

    #Make the point be chosen
    #Say the point chosen is (2,1) - need a function that says this cannot be a move because it is occupied
    
    if isSafe(matrix,neighbor_list[3]) == True:
        print('Validated')
        #Now need to update the thing
    else:
        print('Not valid')

    #Update the array so that the move as been made
    #Do the same for the rest of the 2 elements
    

if __name__ == '__main__':
    main()
