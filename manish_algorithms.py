import itertools

# describing a basic hill climbing algorithm


def hillclimb(L, final):
    Lseen = {}  # list of already checked states
    curr = list(L)[0]  # the first element in L
    # this is all more or less psuedocode until I figure out how it's actually supposed to look
    for i in L:
        if i not in Lseen:  # only check if it's new
            if curr != final:  # presumably 'final' is passed to this function
                # if 'curr' isn't the answer, put it in Lseen
                Lseen.update(curr)
                # curr's children need to go in the parentheses below, must figure out how 'CreateChildren' works
                L.update(sorted(CreateChildren(curr)))
                L.pop(curr)  # remove curr from L
                hillclimb(L, final)  # unsure if this recursive function works
            else:
                return curr  # if it is the answer then return it and stop immediately
                exit(1)  # don't know what the actual exit function is


# this one is mostly the same as before, but the sorting happens on all of L instead of only curr's children
def bestfirst(L, final):
    Lseen = {}  # list of already checked states
    curr = list(sorted(L))[0]  # the first element in the sorted L
    for i in L:
        if i not in Lseen:
            if curr != final:
                Lseen.update(curr)
                L.update(CreateChildren(curr))
                L.pop(curr)
                hillclimb(L, final)
            else:
                return curr
                exit(1)
