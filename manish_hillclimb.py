import itertools


def hillclimb(L, final):
    Lseen = {}
    curr = list(L)[0]
    for i in L:
        if i not in Lseen:
            if curr != final:
                Lseen.update(curr)
                L.update()  # curr's children need to go in the parentheses
                hillclimb(L, final)
            else:
                return curr
