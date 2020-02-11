from queue import PriorityQueue
import numpy as np
import itertools
from itertools import product
from random import randint

unique_val = []
INITIAL_1 = (3, 0)
INITIAL_2 = (0, 0)
INITIAL_3 = (1, 1)
FINAL_1 = [2, 1]
FINAL_2 = [0, 1]
FINAL_3 = [2, 2]

FINAL_POINTS = (FINAL_1, FINAL_2, FINAL_3)


class State(object):
		def __init__(self, start=0, goal=0):
				self.children = []
				self.start = start
				self.goal = goal

		def compareStates(self, goal = 0):
			goal = self.goal
			if self.start.all() == goal.all():
				return True

		def makePlayers(self):
			# find unique values on board (ex: 1, 2, 3)
			# ignores 0, which symbolizes empty spaces
			for sublist in self.start:
				for val in sublist:
					if val not in unique_val:
						unique_val.append(val)
				unique_val.remove(0)
			return unique_val

		def isSafe(self, point):
				# The board being the matrix
				# The point being the point chosen from the path(s)
				print('Chosen point to move to is: ', point)
				x = point[0]
				y = point[1]
				print(self.start[x][y])
				if self.start[x][y] != 0:
						return False
				else:
						return True

		def GetDistPerm(self, perm):
			dist_list = []
			sum_list = []
			for i in range(len(perm)):
				for k in range(len(unique_val)):
					#print(perm[i][k], 'vghvhghvhhgvh')
					#print(FINAL_POINTS[k][0], "FINAL POINTS K 0")
					x_val = abs(perm[i][k][0] - FINAL_POINTS[k][0])
					#print(perm[i][k][0], 'permIK0')
					#print(x_val, 'X VAL')
					# counts number of horizontal steps (columns)
					y_val = abs(perm[i][k][1] - FINAL_POINTS[k][1])
					#print(perm[i][k][1], 'permIK1')
					#print(y_val, 'Y VAL')
					# calculates total distance
					distance = y_val + x_val
					# appends distances to list
					dist_list.append(distance)

			j=0
			while j < len(dist_list):
				sum_list.append(sum(dist_list[j:j+3]))
				j+=3

			return sum_list #Returns the distance of every board, so [6,6] the first 6 is distance of board 1 and the other of board 2

		def neighbors(self, index):
				print('The point is at: ', index)
				N = len(index)
				for relative_index in product((-1, 0, 1), repeat=N):
						if not all(i == 0 for i in relative_index):
								everyone = list(i + i_rel for i, i_rel in zip(index, relative_index))
								yield everyone

		def maxBoardLength(self):
				max_rows = len(self.start)
				max_columns = max(map(len, self.start))
				return max_rows, max_columns

		def removeCheats(self, neighborList, point):
				updated_list = []
				max_rows, max_columns = self.maxBoardLength()
				for i in range(len(neighborList)):
						# Removes the diagonals
						if abs(sum(neighborList[i]) - sum(point)) != 2 and abs(sum(neighborList[i]) - sum(point)) != 0:
								# Removes the negatives
								# print(neighborList[i][0], 'this is the neighbor list')
								if neighborList[i][0] >= 0 and neighborList[i][1] >= 0:
										# Takes care of the out of grid
										if neighborList[i][0] < max_rows and neighborList[i][1] < max_columns:
												# This code checks what available spots are on the grid, and will only add open spaces.
												# Will not allow spaces already occupied by 1, 2, or 3 to be added to the list
												if self.start[neighborList[i][0]][neighborList[i][1]] == 0:
														updated_list.append(neighborList[i])
				return updated_list

		def getIndices(self, pair_points):
				print(pair_points, "pair_points")
				points_to_move_with = []
				individual_points = []
				for i in range(len(pair_points)):
						for k in range(len(pair_points) - 1):
								print(pair_points[i][0][k])
								if pair_points[i][0][k] not in FINAL_POINTS:
										points_to_move_with.append(pair_points[i][0][k])
								# pair_points[i][0][k].append(points_to_go_to)

				tupled = list()
				for lk in range(len(points_to_move_with)):
					tupled.append(tuple(points_to_move_with[lk]))
				print(tupled, "_points2__")

				return tupled


		def allPossibleBoards(self, parent_and_child):
			combos = list(product(*(parent_and_child.values())))
			# ELIMINATE MOVES THAT WOULD MOVE 2 NUMBERS TO THE SAME BLOCK
			# Store legal moves in list names "perm". Perm stands for "permutation".
			perm = []
			boardKey_distVal = dict()
			for k in range(0, len(combos)):
					if combos[k][0] != combos[k][1] and combos[k][0] != combos[k][2] and combos[k][1] != combos[k][2]:
							perm.append(combos[k])
			print(perm, "perm")
			#testinger = self.GetDistPerm(perm)
			distances_of_points_on_boards = self.GetDistPerm(perm) #The list of the distances!
			#print(distances_of_points_on_boards)
			#print(testinger[0:3], 'testinger123')
			#print(testinger[3:6], 'testinger456')

			# THE CODE DOWN HERE IS NOW FINISHED
			i = 0
			while i < len(perm):
				temp = self.start.copy()
				#print(temp, 'absolute board is the passed in')
				temp[perm[i][0][0]][perm[i][0][1]] = 1
				temp[perm[i][1][0]][perm[i][1][1]] = 2
				temp[perm[i][2][0]][perm[i][2][1]] = 3
				print(temp, 'absolute after the changes')
				self.children.append(temp)
				#print(self.children)
				i = i+1

			#Need to convert the list back to numpy array
			npp = np.asarray(self.children)

			return npp, distances_of_points_on_boards


		# player 1 start is (0,0) --> 2,1
		def CreateChildren(self):
				# 1- get the players that can move, i.e., not the end state
				# 2- once the players are retrieved, get the available points to move
				# 3- validate all the points at the same time


				pair_points = [[], [], []]  # 3 b/c there are only three players
				parent_child = {}
				players = unique_val

				for i in players:
						print(i)
						have_sx = np.where(self.start == i)  # List of every coordinate that is 0,1,2,3,....
						have_sx = np.asarray(have_sx).T.tolist()
						print(have_sx, 'have_sx')
						pair_points[i - 1].append(have_sx)
						print(pair_points[i - 1][0], 'FOR PLAYER', i)

				#[x,y] not (x,y)

				print(pair_points, 'pair points!!!')
				# Converts the points to indices which can then be passed on to the getNeighbors function and return the neighbors in order to determine children.
				res = self.getIndices(pair_points)  # players/indices that can be moved

				# Saves the parent and its children to a dictionary
				for j in range(len(res)):
						neigh_list = list(self.neighbors(res[j]))
						moves = self.removeCheats(neigh_list, res[j])
						parent_child.update({res[j]: moves})
				print(parent_child, 'parent child')

				children_boards, distances = self.allPossibleBoards(parent_child)
				

				#Using only to see the boards
				for i in range(0, len(children_boards)):
					print('\n', 'Board', i+1, '\n', children_boards[i], end = " ", flush=True)
					print('Their distances', distances[i])
					print('\n')


				return children_boards, distances

class HillSolver: #EVAL FUNCTION SHOULD BE IN THIS CLASS BECAUSE IT IS DIFFERENT FOR EVERY ALGORITHM!

		#Eval function: based of list of heruistics assign numers to each board where a high number means a better bord
		#Make sure the Lseen stuff for the points
		#If it finds a solution but not all the blocks are filled in, go back and pick another way. HAS to match the anser!! Yeah you connected them, but you used the wrong formula.


		def __init__(self, start, goal):
				self.path = [] #this self.path should include the intial board and the children it chooses to go to.
				self.start = start
				self.goal = goal


		def evaluate(self, test):
			tester = randint(0,10)
			print(tester, 'asdasdasdasd')
			return tester		
			
		def HillClimbing(self):
			starting_state = State(self.start, self.goal)
			
			#print(starting_state, 'sstatatt')
			players = starting_state.makePlayers()
			#current = boards #the current state are the boards that are passed in
			#expanded_states = 0
			#viewed_states = 0
			#next_states, next_distances = starting_state.CreateChildren()
			
			#starting_state = next_states[0]
			#starting_state = State(next_states[0], self.goal)
			#players = starting_state.makePlayers()
			#next_states = starting_state.CreateChildren()
			#while True:
				#next_states, next_distances = starting_state.CreateChildren()
				
			
			while True:
				next_states, next_distances = starting_state.CreateChildren() #returns the two children at first
				if len(next_states) == 0:
					print('no next children')
					break
				else:
					best_next_state = next_states[0] # best next is the first child
					for i in range(len(next_states)): #i = 0,1
						print(next_states[i], 'next state i')
						print(best_next_state, 'best next ')
						if (self.evaluate(next_states[i]) < self.evaluate(best_next_state)): #if the distance of the first child is > than the value of the best next state							
							print('next state < best next state')							
							best_next_state = next_states[i] #Make the best next the next state, which has a lesser value
						elif (self.evaluate(next_states[i]) > self.evaluate(best_next_state)):
							print('next state > best next state')							
							
					print(best_next_state, 'THIS IS THE BEST NEXT STATE')
					starting_state = State(best_next_state, self.goal)
					
				'''else:
					best_next = next_states[0]
					for state in range(len(next_states)):
						if self.evaluate(next_states[state]) > self.evaluate(best_next):
							
						#if self.evaluate(next_states[state]) > self.evaluate(best_next):
							best_next = next_states[state]
					starting_state = next_states
			'''
			
		def solve(self):

				#Call makePlayers to get the number of distinct values
				#getDist would take in unique_val
				#Call CreateChildren() to make the children_boards
				#Here, get the distances of each board using GetDist
				#Have an eval function that chooses which board to make children for
				#Continue

				L_seen = list()
				self.path.append(self.start)

				self.HillClimbing()
				
				#initialState = State(self.start, self.goal)  # 0 because not a parent
				#players = initialState.makePlayers()
				#children_boards, distances = initialState.CreateChildren()

				#Call the eval() function from here
				#self.HillClimbing(children_boards)
				
				
				
				
				#How to add to L seen
				
				#L_seen.append(np.asarray(self.start).T.tolist())

				#LSEEN would have the points it started on. So automatically add the points when the board is passed in

				'''if initialState.isSafe(self.start, moves[1]) == True:
						print('Valid')
				else:
						print('Not valid')'''



if __name__ == '__main__':
		HillStart = np.array([[1, 2, 0, 3],
													[0, 0, 0, 0],
													[0, 1, 0, 3],
													[2, 0, 0, 0]])


		HillGoal = np.array([[1, 2, 2, 2],
												[1, 3, 3, 2],
												[1, 1, 3, 2],
												[2, 2, 2, 2]])

		test = HillSolver(HillStart, HillGoal)
		test.solve()