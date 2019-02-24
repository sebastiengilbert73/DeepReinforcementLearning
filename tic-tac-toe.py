import torch

firstPlayer = 'X'
secondPlayer = 'O'
playerToPlaneIndexDic = {'X': 0, 'O': 1}
positionTensorShape = (2, 1, 3, 3)
moveTensorShape = (1, 1, 3, 3)

class Authority():
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor),
    #   Winner(self, positionTensor)
    #   LegalMovesMask(self, positionTensor)
    def __init__(self):
        pass


    def ThereIs3InARow(self, planeNdx, positionTensor):
        if positionTensor.shape != (2, 1, 3, 3): # (C, D, H, W)
            raise ValueError("Authority.ThereIs3InARow(): The shape of positionTensor ({}) is not (2, 1, 3, 3)".format(positionTensor.shape))
        # Horizontal lines
        for row in range(3):
            theRowIsFull = True
            for column in range(3):
                if positionTensor[planeNdx, 0, row, column] != 1:
                    theRowIsFull = False
            if theRowIsFull:
                return True

        # Vertical lines
        for column in range(3):
            theColumnIsFull = True
            for row in range(3):
                if positionTensor[planeNdx, 0, row, column] != 1:
                    theColumnIsFull = False
            if theColumnIsFull:
                return True

        # Diagonal \
        diagonalBackslashIsFull = True
        for index in range(3):
            if positionTensor[planeNdx, 0, index, index] != 1:
                diagonalBackslashIsFull = False
        if diagonalBackslashIsFull:
            return True

        # Diagonal /
        diagonalSlashIsFull = True
        for index in range(3):
            if positionTensor[planeNdx, 0, index, 2 - index] != 1:
                diagonalSlashIsFull = False
        if diagonalSlashIsFull:
            return True

        # Otherwise
        return  False

    def Winner(self, positionTensor):
        Xwins = self.ThereIs3InARow(playerToPlaneIndexDic['X'], positionTensor)
        Owins = self.ThereIs3InARow(playerToPlaneIndexDic['O'], positionTensor)
        if Xwins:
            return 'X'
        if Owins:
            return 'O'
        else:
            #print ("Authority.Winner(): torch.nonzero(positionTensor) = {}".format(torch.nonzero(positionTensor)))
            if torch.nonzero(positionTensor).size(0) == 9: # All squares are occupied
                return 'draw'
            else:
                return None

    def MoveWithCoordinates(self, currentPositionTensor, player, dropCoordinates):
        if currentPositionTensor.shape != (2, 1, 3, 3): # (C, D, H, W)
            raise ValueError("Authority.MoveWithCoordinates(): The shape of currentPositionTensor ({}) is not (2, 1, 3, 3)".format(currentPositionTensor.shape))
        if player != 'X' and player != 'O':
            raise ValueError("Authority.MoveWithCoordinates(): The player must be 'X' or 'O', not '{}'".format(player))
        if len(dropCoordinates) != 2:
            raise ValueError("Authority.MoveWithCoordinates(): dropCoordinates ({}) is not a 2-tuple".format(dropCoordinates))
        if dropCoordinates[0] < 0 or dropCoordinates[0] > 2 or dropCoordinates[1] < 0 or dropCoordinates[1] > 2:
            raise ValueError("Authority.MoveWithCoordinates(): dropCoordinates entries ({}) are not in the range [0, 2]".format(dropCoordinates))
        if currentPositionTensor[0, 0, dropCoordinates[0], dropCoordinates[1]] != 0 or \
                currentPositionTensor[1, 0, dropCoordinates[0], dropCoordinates[1]] != 0:
            raise ValueError("Authority.MoveWithCoordinates(): Attempt to drop in an occupied square ({})".format(dropCoordinates))
        currentPositionTensor[playerToPlaneIndexDic[player], 0, dropCoordinates[0], dropCoordinates[1]] = 1
        winner = self.Winner(currentPositionTensor)
        return currentPositionTensor, winner

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != (1, 1, 3, 3):
            raise ValueError("Authority.Move(): moveTensor.shape ({}) is not (1, 1, 3, 3)".format(moveTensor.shape))
        numberOfOnes = 0
        dropCoordinates = None
        for row in range(3):
            for column in range(3):
                if moveTensor[0, 0, row, column] == 1:
                    numberOfOnes += 1
                    dropCoordinates = (row, column)
        if numberOfOnes != 1:
            raise ValueError("Authority.Move(): The number of ones in moveTensor ({}) is not one".format(numberOfOnes))
        return self.MoveWithCoordinates(currentPositionTensor, player, dropCoordinates)

    def Display(self, positionTensor):
        if positionTensor.shape != positionTensorShape: # (C, D, H, W)
            raise ValueError("Authority.Display(): The shape of positionTensor ({}) is not (2, 1, 3, 3)".format(positionTensor.shape))
        for row in range(3):
            for column in range(3):
                #occupancy = None
                if positionTensor[playerToPlaneIndexDic['X'], 0, row, column] == 1.0:
                    print (' X ', end='', flush=True)
                elif positionTensor[playerToPlaneIndexDic['O'], 0, row, column] == 1.0:
                    print (' O ', end='', flush=True)
                else:
                    print ('   ', end='', flush=True)
                if column != 2:
                    print ('|', end='', flush=True)
                else:
                    print('') # new line
            if row != 2:
                print ('--- --- ---')

    def LegalMovesMask(self, positionTensor):
        if positionTensor.shape != positionTensorShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionTensor ({}) is not {}".format(
                positionTensor.shape, positionTensorShape))
        legalMovesMask = torch.zeros(moveTensorShape).byte() + 1 # Initialized with ones, i.e legal moves
        for row in range(3):
            for column in range(3):
                if positionTensor[0, 0, row, column] != 0 or positionTensor[1, 0, row, column] != 0:
                    legalMovesMask[0, 0, row, column] = 0
        return legalMovesMask


def main():
    print ("tic-tac-toe.py main()")
    positionTensor = torch.zeros((2, 1, 3, 3))
    #positionTensor[0, 0, 1, 1] = 1
    positionTensor[0, 0, 0, 1] = 1
    positionTensor[0, 0, 2, 0] = 1
    positionTensor[1, 0, 2, 1] = 1
    positionTensor[0, 0, 0, 2] = 1
    positionTensor[0, 0, 2, 2] = 1
    positionTensor[1, 0, 0, 0] = 1
    positionTensor[1, 0, 1, 2] = 1
    positionTensor[1, 0, 1, 1] = 1
    #positionTensor[0, 0, 1, 0] = 1
    ticTacToeAuthority = Authority()
    ticTacToeAuthority.Display(positionTensor)

    winner = ticTacToeAuthority.Winner(positionTensor)
    print ("Winner: {}".format(winner))

    legalMovesMask = ticTacToeAuthority.LegalMovesMask(positionTensor)
    print ("legalMovesMask = {}".format(legalMovesMask))


if __name__ == '__main__':
    main()