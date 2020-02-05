from queue import PriorityQueue
import numpy as np
from itertools import product

unique_val = []
INITIAL_1 = (3,0)
INITIAL_2 = (0,0)
INITIAL_3 = (1,1)
FINAL_1 = [2,1]
FINAL_2 = [0,1]
FINAL_3 = [2,2]

FINAL_POINTS = (FINAL_1, FINAL_2, FINAL_3)

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
				return dist_list, unique_val
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

		#Need to figure out this. All it does it makes it starts with 3
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

		def getIndices(self,pair_points):
				points_to_move_with = []
				individual_points = []
				for i in range(len(pair_points)):
						for k in range(len(pair_points)-1):
								print(pair_points[i][0][k])
								if pair_points[i][0][k] not in FINAL_POINTS:
										points_to_move_with.append(pair_points[i][0][k])
										#pair_points[i][0][k].append(points_to_go_to)
				
				#Put all the points TOGETHER on a list
				for i in range(len(points_to_move_with)):
						for kl in range(len(points_to_move_with)-1):
								individual_points.append(points_to_move_with[i][kl])
				
				indices = list(zip(individual_points, individual_points[1:] + individual_points[:1]))
				del indices[1::2]
				return indices
		

		#player 1 start is (0,0) --> 2,1
		def CreateChildren(self, board):
				# 1- get the players that can move, i.e., not the end state
				# 2- once the players are retrieved, get the available points to move
				# 3- validate all the points at the same time

				children = []
				pair_points = [[],[],[] ] #3 b/c there are only three players
				parent_child = {}
				players = unique_val

				for i in players:
						print(i)
						have_sx = np.where(board == i) #List of every coordinate that is 0
						have_sx = np.asarray(have_sx).T.tolist()
						pair_points[i-1].append(have_sx)
						print(pair_points[i-1][0], 'FOR PLAYER', i)

				#Converts the points to indices which can then be passed on to the getNeighbors function and return the neighbors in order to determine children.
				res = self.getIndices(pair_points) #players/indices that can be moved
				
				#Saves the parent and its children to a dictionary
				for j in range(len(res)):
						neigh_list = list(self.neighbors(res[j]))
						moves = self.removeCheats(board, neigh_list, res[j])
						parent_child.update({res[j]: moves})
				print(parent_child)


class HillSolver:
		def __init__(self, start, goal):
				self.path          = []
				self.visitedQueue  = []
				self.priorityQueue = PriorityQueue()
				self.start         = start
				self.goal          = goal

		def solve(self):
				initialState = State(self.start, 0, self.start, self.goal) #0 because not a parent
				distances, players = initialState.GetDist(self.start) #returns [3,4,2] - the distances of 1,2,3 respectively
				
				#initial_mover = initialState.eval_function(distances, self.start)
				
				#After this, it should all be to create childre. So createChildren() should call all the other functions.
				#print('initial mover: ', initial_mover)
				
				neighbor_list = list(initialState.neighbors(INITIAL_3))
				print(neighbor_list)
				moves = initialState.removeCheats(self.start, neighbor_list, INITIAL_3)
				print('Available moves are: ', moves)
				
				#Make the point chosen in create children
				#Makes a point be chosen. For testing purposes I made it choose (2,1)
				point_chosen = moves[1]
				if initialState.isSafe(self.start, moves[1]) == True:
						print('Valid')
				else:
						print('Not valid')
				
				#tester = {(0,0): [(0,1), (1,1)]}
				#print(tester[(0,0)])
				#for x in tester:
				#    val = tester[x][0]
				#print(val)
				initialState.CreateChildren(self.start)
				

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
