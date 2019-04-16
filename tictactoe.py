import torch
import policy

firstPlayer = 'X'
secondPlayer = 'O'
#playerToPlaneIndexDic = {'X': 0, 'O': 1}
#positionTensorShape = (2, 1, 3, 3)
#moveTensorShape = (1, 1, 3, 3)

class Authority():
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor),
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self)
    #   MoveTensorShape(self)
    #   InitialPosition(self)
    #   SwapPositions(self, positionTensor, player1, player2)
    #   PlayersList(self)
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


def main():
    print ("tic-tac-toe.py main()")

    authority = Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playersList = authority.PlayersList()
    initialPosition = authority.InitialPosition()
    neuralNetwork = neuralNetwork = policy.NeuralNetwork(positionTensorShape,
                                         '[(3, 16), (3, 16), (3, 16)]',
                                         moveTensorShape)
    neuralNetwork.load_state_dict(torch.load('/home/sebastien/projects/DeepReinforcementLearning/outputs/neuralNet_tictactoe_499.pth'))
    """averageReward, winRate, drawRate, lossRate = \
        policy.AverageRewardAgainstARandomPlayer(
                             playersList,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax=True,
                             softMaxTemperature=1.0,
                             numberOfGames=50,
                             moveChoiceMode='ExpectedMoveValuesThroughSelfPlay',
                             numberOfGamesForMoveEvaluation=11)
    
    print ("averageReward = {}; winRate = {}; drawRate = {}; lossRate = {}".format(averageReward, winRate, drawRate, lossRate))
    """
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (0, 0))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (0, 1))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (0, 2))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (1, 0))
    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (1, 1))
    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (1, 2))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[0], (2, 0))
    initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (2, 1))
    #initialPosition, winner = authority.MoveWithCoordinates(initialPosition, playersList[1], (2, 2))

    moveValuesTensor, standardDeviationTensor, legalMovesMask = policy.PositionExpectedMoveValues(
        playersList,
        authority,
        neuralNetwork,
        initialPosition,
        numberOfGamesForEvaluation=11,
        softMaxTemperatureForSelfPlayEvaluation=0.1
        )
    print ("moveValuesTensor = \n{}".format(moveValuesTensor))
    print ("standardDeviationTensor =\n{}".format(standardDeviationTensor))
    print ("legalMovesMask = \n{}".format(legalMovesMask))

    chosenMove = neuralNetwork.ChooseAMove(
        initialPosition,
        playersList[0],
        authority,
        preApplySoftMax=True,
        softMaxTemperature=1.0,

    )
    print ("chosenMove =\n{}".format(chosenMove))

    highestProbabilityMove = neuralNetwork.HighestProbabilityMove(
        initialPosition, playersList[0], authority)
    print ("highestProbabilityMove =\n{}".format(highestProbabilityMove))

    """positionMoveStatisticsList = policy.GenerateMoveStatistics(playersList,
                           authority,
                           neuralNetwork,
                           proportionOfRandomInitialPositions=0.5,
                           maximumNumberOfMovesForInitialPositions=8,
                           numberOfInitialPositions=2,
                           numberOfGamesForEvaluation=11,
                           softMaxTemperatureForSelfPlayEvaluation=1.0
                           )
    print ("main(): positionMoveStatisticsList = {}".format(positionMoveStatisticsList))
    """

if __name__ == '__main__':
    main()