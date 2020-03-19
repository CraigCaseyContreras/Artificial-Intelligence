#How it workds: Generate a list of all possible next steps towards goal from current position
#Store children in Priority Queue based on distance to goal, closest first
#Select closest child and repeat until goal is reached or no more cildren

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
		for i in range(len(self.goal)): #Go thou each letter of the goal
			letter = self.goal[i]
			#Find the index of that letter in our current value and subtract rom where we are at
			#Distance that the letter is from our target 
			try:
				distance += abs(i - self.value.index(letter))
			except:
				distance += abs(i - self.value.find(letter))
		print(self.value, 'has a distance of: ', distance)  
		return distance 
		
   	
	def CreateChildren(self):
		if not self.children: #if there is no children
			for i in range(len(self.goal)-1): #Go through every possiblity of every possible arrangement of the 
				val = self.value
				#Switch the second letter and the first letter of everypair of lettters
				#Then add on the beginning and the end and new arrangements of letters
				val = val[:i] + val[i+1] + val[i] + val[i+2:] 
				print('Child',i+1, ':',val)
				#Store the value we generated in the child
				child = StringState(val, self)
				self.children.append(child)
				#print(self.children, 'self children')
				
		print('---------------------------')
		print('---------------------------')
		#print('---------------------------')
		#print('---------------------------')
		#print('---------------------------')


class HillSolver():
	def __init__(self, start, goal):
		self.start = start
		self.goal = goal
		
		
	def solve(self):
		L_seen = []
		print('Solve from hillsover')
		L_seen.append(self.start)
		
		startState = StringState(self.start, 0, self.start, self.goal)
		startState.CreateChildren()
		#print(startState.children) #scrambled word
		
		for child in startState.children:
			if child.value not in L_seen:
				if child.GetDist() == 0:
					return child.value
				elif child.GetDist < 
					
				
		print(L_seen)
			
		
		
class AStar_Solver:
	def __init__(self, start , goal):
		self.path          = []
		self.visitedQueue  = []
		self.priorityQueue = PriorityQueue()
		self.start         = start
		self.goal          = goal

	def Solve(self):
		print('Solve from A solver')
		startState = StringState(self.start,
											0,
											self.start,
											self.goal)

		count = 0
		i = 0
		#A tuple. Every state is held in startState
		self.priorityQueue.put((0,count,startState))

		#While the path is empty AND while the priorityQueue HAS a size
		while(not self.path and self.priorityQueue.qsize()):
				#Equal to the StartState so the 2 slot of (0,count,startState)
				closestChild = self.priorityQueue.get()[2]
				print(closestChild.__dict__, 'closeChild')
				
				
				#Makes the children for that states
				closestChild.CreateChildren()
				
				#Add child to visited queue
				print(closestChild.value, 'closest child val')
				self.visitedQueue.append(closestChild.value)

				#Go through each child that was created for that state
				for child in closestChild.children:
					#if the child has not been visited OR if the child's value is not in the visited queue
					if child.value not in self.visitedQueue:
						i +=1
						#If the distance is at 0 and does not exist
						
						if not child.dist:
								self.path = child.path
								
								break
						
						self.priorityQueue.put((child.dist,i,child))
						#print(count)
				
		self.visitedQueue.append(self.goal)
		print(self.visitedQueue, 'visited queue')
		if not self.path:
				print("Goal of %s is not possible!" % (self.goal))

		return self.path
			
if __name__ == "__main__":
	
	def scramble(unscrambled):
		''' 
		Scrambles the word(s) in unscrambled such that the first and last letter remain the same,
		but the inner letters are scrambled. Preserves the punctuation.
		See also: http://science.slashdot.org/story/03/09/15/2227256/can-you-raed-tihs
		'''
		import string, random, re
		splitter = re.compile(r'\s')
		words = splitter.split(u''.join(ch for ch in unscrambled if ch not in set(string.punctuation)))
		for word in words:
			if len(word) < 4: continue
			mid = list(word[1:-1])
			random.shuffle(mid)
			scrambled = u'%c%s%c' % (word[0], ''.join(mid), word[-1])
			unscrambled = unscrambled.replace(word, scrambled, 1)
    
		return unscrambled
	
	goal1 = "abcde"
	start11 = scramble(goal1)
	print("Starting...")

	#start_time = time.time()
	#a = AStar_Solver(start11, goal1)
	#a.Solve()
	#print("--- %s seconds ---" % (time.time() - start_time))
	
	b = HillSolver(start11, goal1)
	b.solve()