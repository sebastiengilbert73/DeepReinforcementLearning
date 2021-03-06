import torch
import ast
import gameAuthority
import generateMoveStatistics

firstPlayer = 'X'
secondPlayer = 'O'
#playerToPlaneIndexDic = {'X': 0, 'O': 1}
#positionTensorShape = (2, 1, 3, 3)
#moveTensorShape = (1, 1, 3, 3)

class Authority(gameAuthority.GameAuthority):
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor)
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self)
    #   MoveTensorShape(self)
    #   InitialPosition(self)
    #   SwapPositions(self, positionTensor, player1, player2)
    #   PlayersList(self)
    #   MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString)
    #   Display(self, positionTensor)

    def __init__(self):
        self.playersList= ['X', 'O']
        self.positionTensorShape = (2, 1, 3, 3)
        self.moveTensorShape = (1, 1, 3, 3)
        self.playerToPlaneIndexDic = {'X': 0, 'O': 1}


    def ThereIs3InARow(self, planeNdx, positionTensor):
        if positionTensor.shape != self.positionTensorShape: # (C, D, H, W)
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

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        Xwins = self.ThereIs3InARow(self.playerToPlaneIndexDic['X'], positionTensor)
        Owins = self.ThereIs3InARow(self.playerToPlaneIndexDic['O'], positionTensor)
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
        if currentPositionTensor.shape != self.positionTensorShape: # (C, D, H, W)
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
        newPositionTensor = currentPositionTensor.clone()
        newPositionTensor[self.playerToPlaneIndexDic[player], 0, dropCoordinates[0], dropCoordinates[1]] = 1
        winner = self.Winner(newPositionTensor, player)
        return newPositionTensor, winner

    def MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString):
        dropCoordinatesTuple = ast.literal_eval(dropCoordinatesAsString)
        print ("MoveWithString(): dropCoordinatesTuple = {}".format(dropCoordinatesTuple))
        return self.MoveWithCoordinates(currentPositionTensor, player, dropCoordinatesTuple)

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
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

    """def MoveInPlace(self, currentPositionTensor, player, moveTensor):
        dropCoordinates = self.DropCoordinates(moveTensor)
        if currentPositionTensor[0, 0, dropCoordinates[0], dropCoordinates[1]] == 1 or \
            currentPositionTensor[0, 0, dropCoordinates[0], dropCoordinates[1]] == 1:
            raise ValueError("MoveInPlace(): Attempt to drop in {} while it is already occupied")
        currentPositionTensor[self.playerToPlaneIndexDic[player], 0, dropCoordinates[0], dropCoordinates[1]] = 1
        winner = self.Winner(currentPositionTensor, player)
        return winner
    """

    def DropCoordinates(self, moveTensor):
        numberOfOnes = 0
        dropCoordinates = None
        for row in range(3):
            for column in range(3):
                if moveTensor[0, 0, row, column] == 1:
                    numberOfOnes += 1
                    dropCoordinates = (row, column)
        if numberOfOnes != 1:
            raise ValueError("Authority.DropCoordinates(): The number of ones in moveTensor ({}) is not one".format(numberOfOnes))
        return dropCoordinates


    def Display(self, positionTensor):
        #print ("Authority.Display(): positionTensor.shape = \n{}".format(positionTensor.shape))
        if positionTensor.shape != self.positionTensorShape: # (C, D, H, W)
            raise ValueError("Authority.Display(): The shape of positionTensor ({}) is not (2, 1, 3, 3)".format(positionTensor.shape))
        for row in range(3):
            for column in range(3):
                #occupancy = None
                if positionTensor[self.playerToPlaneIndexDic['X'], 0, row, column] == 1.0:
                    print (' X ', end='', flush=True)
                elif positionTensor[self.playerToPlaneIndexDic['O'], 0, row, column] == 1.0:
                    print (' O ', end='', flush=True)
                else:
                    print ('   ', end='', flush=True)
                if column != 2:
                    print ('|', end='', flush=True)
                else:
                    print('') # new line
            if row != 2:
                print ('--- --- ---')

    def LegalMovesMask(self, positionTensor, player):
        if positionTensor.shape != self.positionTensorShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionTensor ({}) is not {}".format(
                positionTensor.shape, self.positionTensorShape))
        legalMovesMask = torch.zeros(self.moveTensorShape).byte() + 1 # Initialized with ones, i.e legal moves
        for row in range(3):
            for column in range(3):
                if positionTensor[0, 0, row, column] != 0 or positionTensor[1, 0, row, column] != 0:
                    legalMovesMask[0, 0, row, column] = 0
        return legalMovesMask

    def PositionTensorShape(self):
        return self.positionTensorShape

    def MoveTensorShape(self):
        return self.moveTensorShape

    def InitialPosition(self):
        initialPosition = torch.zeros(self.positionTensorShape)
        return initialPosition

    def SwapPositions(self, positionTensor, player1, player2):
        player1PlaneNdx = self.playerToPlaneIndexDic[player1]
        player2PlaneNdx = self.playerToPlaneIndexDic[player2]
        swappedPosition = positionTensor.clone()
        swappedPosition[player1PlaneNdx] = positionTensor[player2PlaneNdx]
        swappedPosition[player2PlaneNdx] = positionTensor[player1PlaneNdx]
        return swappedPosition

    def PlayersList(self):
        return self.playersList

    def RaiseAnErrorIfNoLegalMove(self):
        return True


def main():
    import multiprocessing

    print ("tic-tac-toe.py main()")
    import moveEvaluation.ConvolutionStack
    import time

    authority = Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playersList = authority.PlayersList()
    initialPosition = authority.InitialPosition()
    neuralNetwork = moveEvaluation.ConvolutionStack.Net(positionTensorShape,
                                         [(3, 16), (3, 16), (3, 16)],
                                         moveTensorShape)
    neuralNetwork.load_state_dict(torch.load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,3,3)_[(3,16),(3,16),(3,16)]_(1,1,3,3)_tictactoe_295.pth'))

    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (0, 0))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (0, 1))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (0, 2))
    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (1, 0))
    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (1, 1))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (1, 2))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (2, 0))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (2, 1))
    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (2, 2))

    proportionOfRandomInitialPositions = 0.0
    numberOfMovesForInitialPositionsMinMax = (0, 5)
    numberOfInitialPositions = 4
    numberOfGamesForEvaluation = 31
    softMaxTemperatureForSelfPlayEvaluation = 0.3
    epsilon = 0
    depthOfExhaustiveSearch = 1
    chooseHighestProbabilityIfAtLeast = 1.0

    start_time = time.time()


    outputs = generateMoveStatistics.GenerateMoveStatisticsMultiprocessing(
        playersList,
        authority,
        neuralNetwork,
        proportionOfRandomInitialPositions,
        numberOfMovesForInitialPositionsMinMax,
        numberOfInitialPositions,
        numberOfGamesForEvaluation,
        softMaxTemperatureForSelfPlayEvaluation,
        epsilon,
        depthOfExhaustiveSearch,
        chooseHighestProbabilityIfAtLeast,
        additionalStartingPositionsList=[],
        numberOfProcesses=4
    )

    end_time = time.time()
    exectutionTime = end_time - start_time
    print ("outputs = {}".format(outputs))
    print ("executionTime = {}".format(exectutionTime))


if __name__ == '__main__':
    main()