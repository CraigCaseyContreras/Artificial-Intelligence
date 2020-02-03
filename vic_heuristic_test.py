import numpy as np

# Our initial flow free state
matrix = np.array([[1, 2, 0, 0],
                   [0, 3, 0, 0],
                   [0, 1, 3, 0],
                   [2, 0, 0, 0]])

unique_val = []  # empty list. will hold unique values from matrix

# function makes sure we don't step off the board
# I haven't actually used this anywhere....
def isSafe(board, x, y):
    dim = np.size(board,1)
    if 0 <= x < dim and 0 <= y < dim:
        return True

    return False


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
    start = np.argmin(d_list)
    # replace the value at index 2 with 100, so that next time we search for the minimum in the list, it won't pick the
    # value at index 2 again
    d_list[start] = 100
    # the d_list is now [3, 4, 100]. Therefore the next minimum is the value 3 at index zero. Index zero corresponds
    # with the pair of ones on the board.
    print(d_list)
    start = start + 1


eval_function(cost_function(matrix), matrix)
