import numpy as np
from .perm_utils import isPermutation, getRandomPermutation, getIdentityPermutation, __getPermutations
class Permutation:
    """The Permutation class enables the creation, management and operations of permutation-coded integer vectors."""
    
    def __init__(self, *args):
        """Constructor function. """
        if len(args) == 1:
            if isinstance(args[0],list) or isinstance(args[0],np.ndarray):
                if isPermutation(args[0]):
                    # save permutation
                    self.perm = args[0]
                else:
                    raise NameError('The vector indicated does not comply with permutation condition. Permutations need to be specified from 0 to n-1.')
            elif isinstance(args[0],int):
                # create the identity permutation with the specified size
                self.perm=getIdentityPermutation(args[0])
        else:
            # creates a random permutation of size 10 by default.
            self.perm=getRandomPermutation(10)
    
    def printPerm(self):
        """Prints the permutation"""
        print("Permutation: ",self.perm)

    def inverse(self):
        """It returns a Permutation class object with the inverse permutation."""
        permu=np.argsort(self.perm)
        return Permutation(permu)

    def composeWith(self, permutation2):
        """Given a Permutation class object of the same permutation size as the current one, then it returns a Permutation class object with the composition of both permutations."""
        if type(permutation2) is Permutation and len(permutation2.perm) == len(self.perm):
            return Permutation(self.perm[permutation2.perm])
        else:
            raise NameError(
                'The vector indicated is not a Permutation object with a permutation of the same size of the actual one.')

    def setRandomPermutation(self, size):
        """Resets the current permutation with a new random one of the predefined size."""
        self.perm = getRandomPermutation(size)

def getPermutationsFromScores(scores,perm_type="ordering",decreasing=False):
    """This function, given a set of scores, creates the corresponding permutations."""
    perm_list=__getPermutations(scores,perm_type,decreasing)
    object_list = []
    for x in perm_list:
        object_list.append(Permutation(x))
    return object_list

