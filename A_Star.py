import numpy as np
import itertools
from itertools import permutations, combinations

scrambledword = "Cehn3roatry4"
finalword = "Craythorne34"
# scrambledword = "otga"
# finalword = "goat"
lseen = []
count = 0

# COUNTS NUMBER OF STEPS (i.e. every time createchildren() is called
def counter():
	global count
	count += 1

def getdists(sw):
		# returns a list in positional order of the distances the scrambled letters are from their final positions
		dist = 0
		for i in sw:
				# print(sw.index(i), fw.index(i))
				dist += abs(sw.index(i)-finalword.index(i))
		# Distance formula updated to include number of steps
		a_star_dist = dist + count
		return a_star_dist


def createchildren(cw):
	counter()
	childlist = []
	val = cw
	for i in range(len(cw)-1):
			val = val[:i] + val[i+1] + val[i] + val[i+2:]
			childlist.append(val)
	# print(childlist, count)
	return childlist


def a_star(cw):
	lseen.append(cw)
	sortedlist = sorted(list(createchildren(cw)), key=getdists)
	#print("LSeen: ", lseen)
	#print("list of permutations: ", sortedlist)
	#print(sortedlist,'sorted list')
	for perm in sortedlist:
			if perm not in lseen:
					# print(perm, getdists(perm))
					if perm == finalword:
							print("done")
							return perm
							# exit(1)
					else:
							return a_star(perm)


print("final answer:", a_star(scrambledword))
print("final LSeen: ", lseen, 'of length', len(lseen))
