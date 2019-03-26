import torch
import numpy
import policy


class Authority():
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor), # return currentPositionTensor, winner
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self)
    #   MoveTensorShape(self)
    #   InitialPosition(self)
    #   SwapPositions(self, positionTensor, player1, player2)
    #   PlayersList(self)
    def __init__(self, target, maximumPlayedValue):
        self.target = target
        self.maximumPlayedValue = maximumPlayedValue
        self.moveTensorShape = (1, maximumPlayedValue, 1, 1)
        self.positionTensorShape = (1, target + 1, 1, 1)

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
            raise ValueError("Authority.Move(): moveTensor.shape ({}) != self.moveTensorShape ({})".format(moveTensor.shape, self.moveTensorShape))
        if currentPositionTensor.shape != self.positionTensorShape:
            raise ValueError("Authority.Move(): currentPositionTensor.shape ({}) != self.positionTensorShape ({})".format(currentPositionTensor.shape, self.positionTensorShape))
        if torch.nonzero(currentPositionTensor).size(0) != 1:
            raise ValueError("Authority.Move(): There should be a single non-zero value in currentPositionTensor. torch.nonzero(currentPositionTensor).size(0) = {}".forma(torch.nonzero(currentPositionTensor).size(0)))
        if torch.nonzero(moveTensor).size(0) != 1:
            raise ValueError("Authority.Move(): There should be a single non-zero value in moveTensor. torch.nonzero(moveTensor).size(0) = {}".format(torch.nonzero(moveTensor).size(0)))
        # Get the current sum
        currentSum = self.CurrentSum(currentPositionTensor)

        # Get the played value
        playedValue = self.PlayedValue(moveTensor)

        newSum = currentSum + playedValue
        if newSum > self.target:
            raise ValueError("Authority.Move(): The current sum is {} and the played value is {}. The sum can't exceed {}".format(currentSum, playedValue, self.target))

        #newPositionTensor = torch.zeros(self.positionTensorShape)
        newPositionArr = numpy.zeros(self.positionTensorShape)#, dtype=numpy.float64)
        newPositionArr[0, newSum, 0, 0] = 1
        newPositionTensor = torch.from_numpy(newPositionArr).float()
        return newPositionTensor, self.Winner(newPositionTensor, lastPlayerWhoPlayed=player)

    def CurrentSum(self, currentPositionTensor):
        currentSum = 0
        for index in range(self.positionTensorShape[1]):
            if currentPositionTensor[0, index, 0, 0] > 0:
                currentSum = index
        return currentSum

    def SetSum(self, sum):
        if sum < 0 or sum > self.target:
            raise ValueError("Authority.SetSum(): The sum ({}) is out of the range [0, {}]".format(sum, self.target))
        #positionTensor = torch.zeros(self.positionTensorShape)
        positionArr = numpy.zeros(self.positionTensorShape)#, dtype=numpy.float64)
        positionArr[0, sum, 0, 0] = 1
        return torch.from_numpy(positionArr).float()

    def PlayedValue(self, moveTensor):
        playedValue = 0
        for index in range(self.moveTensorShape[1]):
            if moveTensor[0, index, 0, 0] > 0:
                playedValue = index + 1
        return playedValue

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        if positionTensor[0, self.target, 0, 0] > 0:
            return lastPlayerWhoPlayed
        return None

    def LegalMovesMask(self, positionTensor, player):
        currentSum = self.CurrentSum(positionTensor)
        #legalMovesMask = torch.zeros(self.moveTensorShape).byte()
        legalMovesMask = numpy.zeros(self.moveTensorShape)#, dtype=uint8)
        for moveNdx in range(self.moveTensorShape[1]):
            if currentSum + moveNdx + 1 <= self.target:
                legalMovesMask[0, moveNdx, 0, 0] = 1
        return torch.from_numpy(legalMovesMask).float()

    def PositionTensorShape(self):
        return self.positionTensorShape

    def MoveTensorShape(self):
        return self.moveTensorShape

    def InitialPosition(self):
        #initialPositionTensor = torch.zeros(self.positionTensorShape)
        initialPositionArr = numpy.zeros(self.positionTensorShape)#, dtype=numpy.float64)
        initialPositionArr[0, 0, 0, 0] = 1
        return torch.from_numpy(initialPositionArr).float()

    def SwapPositions(self, positionTensor, player1, player2):
        # Nothing to do
        return positionTensor

    def MoveWithInteger(self, currentPositionTensor, player, valueAsInteger):
        if valueAsInteger < 1 or valueAsInteger > self.maximumPlayedValue:
            raise ValueError("Authority.MoveWithInteger(): The value ({}) is out if the range [1, 10]".format(valueAsInteger))
        #moveTensor = torch.zeros(self.moveTensorShape)
        moveArr = numpy.zeros(self.moveTensorShape)#, dtype=numpy.float64)
        moveArr[0, valueAsInteger - 1, 0, 0] = 1
        return self.Move(currentPositionTensor, player, torch.from_numpy(moveArr).float())

    def PlayersList(self):
        playersList = []
        playersList.append('Player1')
        playersList.append('Player2')
        return playersList

    def MaximumPlayValue(self):
        return self.maximumPlayedValue

    def Target(self):
        return self.target

def main():
    print ("reachX.py main()")

    maximumPlayedValue = 3
    target = 16
    authority = Authority(target, maximumPlayedValue)

    neuralNetwork = policy.NeuralNetwork(authority.PositionTensorShape(),
                                         '[(15, 1, 1, 16), (15, 1, 1, 16), (15, 1, 1, 16)]',
                                         authority.MoveTensorShape())
    neuralNetwork.load_state_dict(torch.load("/home/sebastien/projects/DeepReinforcementLearning/neuralNet_108.pth", map_location=lambda storage, location: storage))

    for sum in range(target):
        positionTensor = authority.SetSum(sum)
        probabilitiesTensor, value = neuralNetwork(positionTensor.unsqueeze(0))
        print ("Sum = {}; ".format(sum), end='', flush=True)
        for playedValue in range(maximumPlayedValue):
            print ("{}\t".format(probabilitiesTensor[0, 0, playedValue, 0, 0]), end='', flush=True)
        print("\t{}".format(value.item()))
        """print ("*** Sum = {} ***".format(sum))
        print ("probabilitiesTensor = \n{}".format(probabilitiesTensor))
        print ("value = {}".format(value))
        """

    """positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playersList = authority.PlayersList()

    testPosition = authority.SetSum(1)
    probabilitiesTensor, value = policy.ProbabilitiesAndValueThroughSelfPlay(
        playersList,
        authority,
        None, # Do random moves
        testPosition,
        numberOfGamesForEvaluation=10,
        preApplySoftMax=True,
        softMaxTemperature=0.1,
        numberOfStandardDeviationsBelowAverageForValueEstimate=0.0
    )
    print ("main(): probabilitiesTensor =\n{}".format(probabilitiesTensor))
    print ("main(): value = {}".format(value))
    """
"""
    position = authority.InitialPosition()
    moveNdx = 0
    winner = None


    while winner is None:
        # Ask the player to choose a move
        player = playersList[moveNdx % 2]
        inputNbr = int(input ("Player {}: Sum = {}. Choose a number [1, {}]: ".format(player, authority.CurrentSum(position), maximumPlayedValue)))
        position, winner = authority.MoveWithInteger(position, player, inputNbr)

        moveNdx += 1
        # Swap the players
        position = authority.SwapPositions(position, playersList[0], playersList[1])


    print ("winner = {}".format(winner))
"""

if __name__ == '__main__':
    main()