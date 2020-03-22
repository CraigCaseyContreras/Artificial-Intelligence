import numpy as np
import itertools
from itertools import permutations, combinations

scrambledword = "rcae"
finalword = "care"
lseen = []


def getdists(sw):
    # returns a list in positional order of the distances the scrambled letters are from their final positions
    dist = 0
    for i in sw:
        # print(sw.index(i), fw.index(i))
        dist += abs(sw.index(i)-finalword.index(i))
    print(dist)
    return dist


def createchildren(cw):
    childlist = []
    for i1, i2 in combinations(range(len(cw)), 2):
        newword = list(cw)
        newword[i1], newword[i2] = newword[i2], newword[i1]
        childlist.append(''.join(newword))
    print(childlist)    
    return childlist


def evalhillclimb(cw):
    lseen.append(cw)
    sortedlist = sorted(list(createchildren(cw)), key=getdists)
    #print("LSeen: ", lseen)
    print("list of permutations: ", sortedlist)
    print(sortedlist,'sorted list')
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
print("final LSeen: ", lseen)
