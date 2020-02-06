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
                L.update()  # curr's children need to go in the parentheses
                hillclimb(L, final)  # unsure if this recursive function works
            else:
                return curr  # if it is the answer then return it and stop immediately
                exit(1)  # don't know what the actual exit function is
