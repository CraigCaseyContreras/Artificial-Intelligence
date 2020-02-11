import numpy as np
import itertools
from itertools import permutations, combinations

scrambledword = "Cehn3roatry4"
finalword = "Craythorne34"
lseen = []


def getdists(sw):
		# returns a list in positional order of the distances the scrambled letters are from their final positions
		dist = 0
		for i in sw:
				# print(sw.index(i), fw.index(i))
				dist += abs(sw.index(i)-finalword.index(i))
		#print(dist)
		return dist


def createchildren(cw):
		childlist = []
		val = cw
		for i in range(len(cw)-1):	
				val = val[:i] + val[i+1] + val[i] + val[i+2:] 
				childlist.append(val)
		return childlist


def evalhillclimb(cw):
		lseen.append(cw)
		sortedlist = sorted(list(createchildren(cw)), key=getdists)
		#print("LSeen: ", lseen)
		#print("list of permutations: ", sortedlist)
		#print(sortedlist,'sorted list')
		for perm in sortedlist:
				if perm not in lseen:
						# print(perm, getdists(perm))
						if getdists(perm) == 0:
								# print("done")
								return perm
								# exit(1)
						else:
								return evalhillclimb(perm)


print("final answer:", evalhillclimb(scrambledword))
print("final LSeen: ", lseen, 'of length', len(lseen))
