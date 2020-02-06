#!/usr/bin/env python

from queue import PriorityQueue

class State(object):
	def __init__(self, value, parent, start = 0, goal = 0):
		self.children = [];
		self.parent = parent;
		self.value = value;
		self.dist = 0;
		if parent:
			self.path = parent.path[:];
			self.path.append(value);
			self.start = parent.start;
			self.goal = parent.goal;
		else:
			self.path = [value];
			self.start = start;
			self.goal = goal;
	
	#This should just get the distances of the numbers to its final state??
	def GetDist(self):
		pass

	#This should create all the children
	def CreateChildren(self):
		pass



#We need to do hill-climb, Best-first, beam search, A* search, annleaing??


#### Hill climb algorithm#
#Step 1 : Evaluate the Initial state. If the first state is a goal state then stop and return success. Else, make the Initial state as Current state. 
#Step 2 : Loop until the Solution state is found or if there are no new operators present which can be applied to Current state.
#a) Select a state that has not yet been applied to the Current state and then apply it to produce a New state.
#b) Perform these steps to evaluate New state:
    #i. If the Current state is the Goal state, then Stop and return success.
    #ii. If the New state is better than the Current state, then make it the Current state and proceed further.
    #iii. If the New state is not better than the current state, then continue the loop until a solution is found.
#Step 3 : Exit.
