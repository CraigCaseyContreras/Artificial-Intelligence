#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:52:07 2019

@author: german
"""

import numpy as np
import pdb
import random
import collections
from queue import PriorityQueue
import heapq
import queue as Q


class SearchAI:
		def __init__(self,Istate,goal_state, numRows, numCols):
				self.Istate=Istate
				self.numRows = numRows
				self.numCols = numCols
				self.goal_state = goal_state
		## This function finds for the child states
		def toggle(self,parent_state, numRows, numColumns):
				temp_state = np.copy(parent_state)
				child_states = []
				action = list()
				for i in range(numRows):
						for j in range(numColumns):
										temp_state[i,:] = np.logical_not(parent_state[i,:]) 
										#pdb.set_trace()
										temp_state[:,j] = np.logical_not(parent_state[:,j]) 
										#pdb.set_trace()
										child_states.append(temp_state)
										temp_state = np.copy(parent_state)
										action.append([i,j])
				return child_states, action
    
		# get representation for state
		def getState(self,array, numRows, numColumns):
				value = ""
				for i in range(numRows):
						for j in range(numColumns):
								value = value + str(array[i,j])
				print(value, 'value from get state!!!!')
				return value
        
		def getArray(self,state, numRows, numColumns):
				array = np.zeros((numRows, numColumns),  dtype=int)
				for i in range(numRows):
						for j in range(numColumns):
								array[i,j] = state[i * numColumns + j]
				return array
    
		def printState(self,state, numRows, numColumns):
				for i in range(numRows):
						row = ""
						for j in range(numColumns):
								index = i * numColumns + j
								row = row + " " + str(state[index])
						print(row)
		# Produce a backtrace of the actions taken to find the goal node, using the 
		# recorded meta dictionary
		def construct_path(self,state, meta):
				action_list = list()
				# Continue until you reach root meta data (i.e. (None, None))
				#pdb.set_trace()
				while meta[state][0] is not None:
						state, action = meta[state]
						action_list.append(action)
				action_list.reverse()
				return action_list
		def printWinningSeqState(self,Istate,actions,numRows, numColumns):
						#winningList = list()
						self.printState(Istate,numRows, numColumns)
						state=Istate
						for idx in actions:
										# debug
										#winningList.append(state)
										####
										array=self.getArray(state, numRows, numColumns)
										tempArray = np.copy(array)
										tempArray[idx[0],:] = np.logical_not(array[idx[0],:]) 
										tempArray[:,idx[1]] = np.logical_not(array[:,idx[1]])
										state = self.getState(tempArray, numRows, numColumns)
										print(state, 'state state')
										print('---------')
										self.printState(state,numRows, numColumns)     
										
							
					
		def evalHill(self, Istate, goal, numRows, numColumns ):
			h = []
			myList = []
			myQueue = Q.PriorityQueue()
			currentArray = self.getArray(Istate, numRows, numColumns)
			goalState = self.getArray(goal, numRows, numColumns)
			print(type(goalState), "type")
			
			Lseen=list()
			#L.append(self.getState(currentArray,numRows,numColumns))
			#L.append(Istate) #0000
			#currentState = L[0]
			flag = 0
			#currentArray = self.getArray(currentState, numRows,numColumns)
			myL = []
			while flag == 0:
				L = []
			#for i in range(7):
				# make children
				print(currentArray, "CURRENT ARRAY")
				childs, action = self.toggle(currentArray, numRows, numColumns)
				print(childs, "CHILDSSSSS")
				for j in childs:
					print(j, "J")
					if self.getState(j, numRows,numColumns) not in Lseen:
						if self.getState(j, numRows,numColumns) == "1111":
							print("FINISHHHHHHHHHHHHHHHHHHH")
							flag = 1
						
						
						print(j, "child")
						kk = self.getState(j, numRows, numColumns)
						print(kk, 'kkkkkkk')
						myL.append(kk)
							
						#myL.append(self.getState(j, numRows, numColumns))
						# sums row's 1 and 2 for each child
						summa = sum(j[0] + j[1])
						print(summa, 'sum of all the board')
						
						L.append([summa, j])
						print(L, 'DAFSDFGasdgA')

				#L.remove(self.getState(currentArray, numRows, numColumns))
				Lseen.append(self.getState(currentArray, numRows, numColumns))
          	
				print(Lseen, 'This is L Seen')
				indices = []
				print(L, 'THIS IS L')
				# pick the child with the highest value
				for ik in range(len(L)):
					indices.append(L[ik][0])
					maxpos = indices.index(max(indices)) #gives the index
				print(maxpos)
            
				#How to save the first one?
				child_to_go_to = L[maxpos][1]
				print('Chosen child: ', child_to_go_to)
				Lseen.append(self.getState(child_to_go_to, numRows, numColumns))
				print(Lseen, 'LSEEEEN')
				print(self.getState(child_to_go_to, numRows, numColumns), 'gfdfggfdgd')
				myL.remove(self.getState(child_to_go_to, numRows, numColumns))
				print(myL, 'MYYYYL')				
			
				
				currentArray = child_to_go_to
				print('-------------------------------')
				print('-------------------------------')
				print('-------------------------------')
		'''
		currentArray2 = child_to_go_to
		print('Next go')
		childs2, action2 = self.toggle(currentArray2, numRows, numColumns)
		print(childs2)
			
			myList2 = []
			for j in childs2:
		if j not in lseen:
			print(j)
			summa2 = sum(j[0] + j[1])
			print(sum(j[0]) + sum(j[1]), 'sum of all the baord')
			#A = list(i, summa)
			myList2.append([summa2, np.asarray(j)])
			#(np.asarray(i).T.tolist(), np.asarray(i).T.tolist())
			#print(type(i))
			indices2 = []
			print(myList2, 'my list after second go!!')
				
				
			child_to_go_to2 = myList2[maxpos][1]
			print('Chosen child: ', child_to_go_to2)
			currentArray3 = child_to_go_to2
			print('Next go')
			childs3, action3 = self.toggle(currentArray3, numRows, numColumns)
			print(childs3)
				#heapq.heappush(h, (summa, childs[i]))
			#print(h)

			'''
				#myQueue.put(i, summa)
			#print(myQueue)
			#print(currentArray, 'current formeval!!!!')
			#print(childs, 'childs from eval!!!!!')
				
		def breadth_search(self,Istate=None,goal_state=None,numRows=None,numColumns=None):
				if Istate == None:
						Istate = self.Istate
				if numRows == None:
						numRows = self.numRows
				if numColumns == None:
						numColumns = self.numCols
				if goal_state == None:
						goal_state = self.goal_state
				## 2-Create Lseen and L
				Lseen = set()
				L = list()
				self.evalHill(Istate, goal_state, numRows, numColumns)
				## 3-Find childrens
				#Istate = getState(np.zeros((numRows, numColumns),  dtype=int), numRows, numColumns)
				#goal_state = getState(np.ones((numRows,numColumns), dtype=int), numRows, numColumns)
				#goal_state = '01010101010101010101'
				print("goal state is")
				#print(goal_state)
				#L.append(Istate)
        
				# a dictionary to maintain meta information (used for path formation)
					# key -> (parent state, action to reach child)
				#meta = dict()
				#meta[Istate] = (None,None)
				#breadth first search
				#1.Let L be list of initial Nodes
				#2. let n be first node in L, if L is empty set, fail
				#3. if n is terminal state, stop
				#4. remove n from L, add to end of L all of n's childrens randomly ordered
				#5. Go to 2
				'''while len(L) > 0:
						currentState = L[0]
						currentArray = self.getArray(currentState, numRows, numColumns)
            
						#print the state
						#print("****************************")
						#print("Current State is")
						#self.printState(currentState, numRows, numColumns)
						#print("****************************")
            
						if currentState == goal_state:
								print('Reach goal state') 
								print("****************************")
								print('Winning sequence of actions:')
								action_list=self.construct_path(currentState, meta) #Prints out the list of actions that lead to a win
								print(action_list)
								self.printWinningSeqState(Istate,action_list,numRows, numColumns)
								print('print winnign state')
								break
						else: 
								if currentState not in Lseen:
										Lseen.add(currentState)
										#print("Length of Lseen", len(Lseen))
										childs, action = self.toggle(currentArray, numRows, numColumns)
                    
										c = list(zip(childs, action))
										random.shuffle(c)
										childs, action = zip(*c)
										print(currentArray, 'currentArray')
										print(childs, 'ajksndjkansdkjasnjk')
										print(action, 'asdknasdk')
                    
										#random.shuffle(childs)
										for i in range(len(childs)):
												state = self.getState(childs[i], numRows, numColumns)
												notInL = state not in L
												notInLseen = state not in Lseen
												if notInL and notInLseen:
														L.append(state)
														meta[state] = (currentState, action[i]) # create metadata for these nodes
								L.remove(currentState) 
				return action_list
    
		## Depth first function
		def depth_search(self,Istate=None,goal_state=None,numRows=None,numColumns=None):
				if Istate == None:
						Istate = self.Istate
				if numRows == None:
						numRows = self.numRows
				if numColumns == None:
						numColumns = self.numCols
				if goal_state == None:
						goal_state = self.goal_state
				## 2-Create Lseen and L
				Lseen = set()
				L = list()
				## 3-Find childrens
				#numRows = 4
				#numColumns = 4
				#Istate = getState(np.zeros((numRows, numColumns),  dtype=int), numRows, numColumns)
				#goal_state = getState(np.ones((numRows,numColumns), dtype=int), numRows, numColumns)
				print("goal state is")
				print(goal_state)
				L.append(Istate)
        
				# a dictionary to maintain meta information (used for path formation)
					# key -> (parent state, action to reach child)
				meta = dict()
				meta[Istate] = (None,None)
				#depth first search
				#1.Let L be list of initial Nodes
				#2. let n be first node in L, if L is empty set, fail
				#3. if n is terminal state, stop
				#4. remove n from L, add to front of L all of n's childrens randomly ordered
				#5. Go to 2
        
				Lseen = set()
				L = list()
        
				L.append(Istate)
				while len(L) > 0:
						currentState = L[0]
						currentArray = self.getArray(currentState, numRows, numColumns)
            
						#print the state
						#print("****************************")
						#print("Current State is")
						#printState(currentState, numRows, numColumns)
						#print("****************************")
            
						if currentState == goal_state:
								print('Reach goal state') 
								print('Winning sequence of actions:')
								action_list=self.construct_path(currentState, meta)
								print(action_list)
								self.printWinningSeqState(Istate,action_list,numRows, numColumns)
								break
						else: 
								if currentState not in Lseen:
										Lseen.add(currentState)
										#print("Length of Lseen", len(Lseen))
										childs, action = self.toggle(currentArray, numRows, numColumns)
										c = list(zip(childs, action))
										random.shuffle(c)
										childs, action = zip(*c)
										#random.shuffle(childs)
										#for array in childs:
										for i in range(len(childs)):
												#state = getState(array, numRows, numColumns)
												state = self.getState(childs[i], numRows, numColumns)
												notInL = state not in L
												notInLseen = state not in Lseen
												if notInL and notInLseen:
														L.insert(0, state)
														meta[state] = (currentState, action[i]) # create metadata for these nodes
								L.remove(currentState)
				return action_list 
    
		'''

if __name__ == '__main__':
	SearchAI = SearchAI('0000', '1111', 2,2)
	SearchAI.breadth_search()
