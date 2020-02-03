import numpy as np

matrix = np.array([[1, 2, 0, 0],
                   [0, 3, 0, 0],
                   [0, 1, 3, 0],
                   [2, 0, 0, 0]])
unique_val = []

def isSafe(board, x, y):
    dim = np.size(board,1)
    if 0 <= x < dim and 0 <= y < dim:
        return True

    return False

def cost_function(board):
    dist_list = []
    for sublist in board:
        for val in sublist:
            if val not in unique_val:
                unique_val.append(val)
        unique_val.remove(0)
    for i in unique_val:
        coord = np.where(board == i)
        coord = np.asarray(coord).T.tolist()
        x_val = abs(coord[0][0] - coord[1][0])
        y_val = abs(coord[0][1] - coord[1][1])
        distance = y_val + x_val
        dist_list.append(distance)
    return dist_list
    # starter = dist_list.index(min(dist_list)) + 1
    # print(starter)


def eval_function(d_list, board):
    # start = d_list.index(min(d_list)) + 1
    start = np.argmin(d_list)
    d_list[start] = 100
    print(d_list)
    start = start + 1


    # d_list.replace(min(d_list))
    # print(d_list)


eval_function(cost_function(matrix), matrix)
