#How it works: Generate a list of all possible next steps towards goal from current position
#Store children in Priority Queue based on distance to goal, closest first
#Select closest child and repeat until goal is reached or no more children

#A* feed in jumble of letters and A* finds the minimal moves to reach goal arrangements of letters
#h(n) is the heuristic path from current node to destination
#g(n) is the path cost from one node to another, so always 2

from queue import PriorityQueue
import time

class State(object):
	def __init__(self, value, parent, start = 0, goal = 0):
		self.children = [] #The children, so the branching stuff
		self.parent = parent #Store the current parent
		self.value = value
		self.dist = 0
		if parent: #Meaning if the parent is plugged in
			self.path = parent.path[:] #copy parent path to our path, so if make changes wont make changes to parent path
			self.path.append(value)
			self.goal = parent.goal
			self.start = parent.start
		else: #Meaning if it is a child
			self.path = [value] #list of objects starting with our current value
			self.start = start
			self.goal = goal
			
	def GetDist(self):
		return
	
	def CreateChildren(self):
		return

class StringState(State):
	def __init__(self, value, parent, start = 0, goal = 0): 
		super(StringState, self).__init__(value, parent, start, goal)
		self.dist = self.GetDist()
   	
	#Check if we have reached our goal
	def GetDist(self):
		if self.value == self.goal:
			print('Reached!!')
			return 0
		distance = 0
		for i in range(len(self.goal)): #Go thru each letter of the goal
			letter = self.goal[i]
			#Find the index of that letter in our current value and subtract rom where we are at
			#Distance that the letter is from our target 
			try:
				distance += abs(i - self.value.index(letter))
			except:
				distance += abs(i - self.value.find(letter))
		print(self.value, 'has a distance of: ', distance+2)  
		return distance +2
		
   	
	def CreateChildren(self, bestChild):
		for i in range(len(self.goal)-1): #Go through every possibility of every possible arrangement of the
			val = list(bestChild)
			#Switch the second letter and the first letter of every pair of lettters
			#Then add on the beginning and the end and new arrangements of letters
			val = val[:i] + val[i+1] + val[i] + val[i+2:]
			print('Child',i+1, ':',val)
			#Store the value we generated in the child
			child = StringState(val, self)
			print(child, "CHILD")
			self.children.append(child)
		print('---------------------------')
		print('---------------------------')
		print('---------------------------')
		print('---------------------------')
		print('---------------------------')
   			
class Best_First_Solver:
	def __init__(self, start , goal):
		self.path          = []
		self.visitedQueue  = []
		self.priorityQueue = []
		self.start         = start
		self.goal          = goal

	def Solve(self):
		# startState = StringState(self.start,
		# 									0,
		# 									self.start,
		# 									self.goal)
		#
		# count = 0
		# #A tuple. Every state is held in startState
		dist = 0
		self.priorityQueue.append([dist, self.start])
		print(self.priorityQueue, "priorityQueue")
		children = []
		flag = 0
		#While the path is empty AND while the priorityQueue HAS a size
		while flag == 0:
				#Equal to the StartState so the 2 slot of (0 ,startState)
				closestChild = self.priorityQueue[0][1]
				print(closestChild, "CLOSEST CHILD")
				#Makes the children for that states
				closestChildKids = CreateChildren(closestChild)
				print(closestChildKids, "CLOSEST CHILD KIDS")
				#Add child to visited queue
				self.visitedQueue.append(closestChild)

				#Go through each child that was created for that state
				for child in closestChildKids:
					#if the child has not been visited OR if the child's value is not in the visited queue
					if child not in self.visitedQueue:
						count +=1
						#If the distance is at 0 and does not exist
						if not child.dist:
								self.path = child.path
								break

						self.priorityQueue.put((child.dist, child))
		self.visitedQueue.append(self.goal)
		print(self.visitedQueue, 'visited queue2')
		if not self.path:
				print("Goal of %s is not possible!" % self.goal)

		return self.path
			
if __name__ == "__main__":
	start1 = "ogta"
	goal1  = "goat"
	print("Starting...")

	start_time = time.time()
	a = Best_First_Solver(start1, goal1)
	print("--- %s seconds ---" % (time.time() - start_time))
	a.Solve()
