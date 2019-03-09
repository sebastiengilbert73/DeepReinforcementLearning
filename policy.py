import argparse
import ast
import torch
import math
import random


class NeuralNetwork(torch.nn.Module):
    def __init__(self, inputTensorSize, bodyStructure, outputTensorSize): # Both input and output tensor sizes must be (C, D, H, W)
        super(NeuralNetwork, self).__init__()
        if len(inputTensorSize) != 4:
            raise ValueError("NeuralNetwork.__init__(): The length of inputTensorSize ({}) is not 4 (C, D, H, W)".format(len(inputTensorSize)))
        if len(outputTensorSize) != 4:
            raise ValueError("NeuralNetwork.__init__(): The length of outputTensorSize ({}) is not 4 (C, D, H, W)".format(len(outputTensorSize)))
        self.inputTensorSize = inputTensorSize;

        if ast.literal_eval(bodyStructure) == [(3, 32), (3, 32)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=inputTensorSize[0], out_channels=32,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=32, out_channels=32,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.lastLayerInputNumberOfChannels = 32
        elif ast.literal_eval(bodyStructure) == [(7, 1, 1, 32), (7, 1, 1, 30)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=inputTensorSize[0], out_channels=32,
                                kernel_size=(7, 1, 1),
                                padding=(3, 0, 0)),
                torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=32, out_channels=30,
                                kernel_size=(7, 1, 1),
                                padding=(3, 0, 0)),
                torch.nn.BatchNorm3d(30, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.lastLayerInputNumberOfChannels = 30
        else:
            raise NotImplementedError("NeuralNetwork.__init__(): Unknown body structure '{}'".format(bodyStructure))

        self.probabilitiesChannelMatcher = torch.nn.Conv3d(in_channels=self.lastLayerInputNumberOfChannels,
                                              out_channels=outputTensorSize[0],
                                              kernel_size=1,
                                              padding=0
                                              )
        self.probabilitiesResizer = torch.nn.Upsample(size=outputTensorSize[-3:], mode='trilinear')
        self.outputTensorSize = outputTensorSize
        self.lastLayerNumberOfFeatures = self.lastLayerInputNumberOfChannels * \
            inputTensorSize[-3] * inputTensorSize[-2] * inputTensorSize [-1]
        self.valueHead = torch.nn.Sequential(
            torch.nn.Linear(self.lastLayerNumberOfFeatures, math.ceil(math.sqrt(self.lastLayerNumberOfFeatures))),
            torch.nn.ReLU(),
            torch.nn.Linear(math.ceil(math.sqrt(self.lastLayerNumberOfFeatures)), 1)
        )

    def forward(self, inputs):
        # Compute the output of the body
        bodyOutputTensor = self.bodyStructure(inputs)
        #print ("NeuralNetwork.forward(): bodyOutputTensor.shape = {}".format(bodyOutputTensor.shape))

        # Move probabilities
        moveProbabilitiesActivation = self.probabilitiesChannelMatcher(bodyOutputTensor)
        #print ("NeuralNetwork.forward(): moveProbabilitiesActivation.shape = {}".format(moveProbabilitiesActivation.shape))
        moveProbabilitiesTensor = self.probabilitiesResizer(moveProbabilitiesActivation)
        #print ("NeuralNetwork.forward(): moveProbabilitiesTensor.shape = {}".format(moveProbabilitiesTensor.shape))

        # Value
        #print ("NeuralNetwork.forward(): bodyOutputTensor.shape = {}".format(bodyOutputTensor.shape))
        #print ("NeuralNetwork.forward(): self.lastLayerNumberOfFeatures = {}".format(self.lastLayerNumberOfFeatures))
        bodyOutputVector = bodyOutputTensor.view(-1, self.lastLayerNumberOfFeatures)
        valueActivation = self.valueHead(bodyOutputVector)
        #print ("NeuralNetwork.forward(): valueActivation = {}".format(valueActivation))
        return moveProbabilitiesTensor, valueActivation

    def ChooseAMove(self, positionTensor, player, gameAuthority, preApplySoftMax=True, softMaxTemperature=1.0):
        rawMoveProbabilitiesTensor, value = self.forward(positionTensor.unsqueeze(0)) # Add a dummy minibatch
        # Remove the dummy minibatch
        rawMoveProbabilitiesTensor = torch.squeeze(rawMoveProbabilitiesTensor, 0)

        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
        normalizedProbabilitiesTensor = NormalizeProbabilities(rawMoveProbabilitiesTensor,
                                                               legalMovesMask,
                                                               preApplySoftMax=preApplySoftMax,
                                                               softMaxTemperature=softMaxTemperature)
        randomNbr = random.random()
        probabilitiesTensorShape = normalizedProbabilitiesTensor.shape
        #print ("NeuralNetwork.ChooseAMove(): normalizedProbabilitiesTensor =\n{}".format(normalizedProbabilitiesTensor))
        runningSum = 0
        chosenCoordinates = None
        for ndx0 in range(probabilitiesTensorShape[0]):
            for ndx1 in range(probabilitiesTensorShape[1]):
                for ndx2 in range(probabilitiesTensorShape[2]):
                    for ndx3 in range(probabilitiesTensorShape[3]):
                        runningSum += normalizedProbabilitiesTensor[ndx0, ndx1, ndx2, ndx3]
                        if runningSum >= randomNbr and chosenCoordinates is None:
                            chosenCoordinates = (ndx0, ndx1, ndx2, ndx3)

        if chosenCoordinates is None:
            raise IndexError("NeuralNetwork.ChooseAMove(): choseCoordinates is None...!???")
        chosenMoveTensor = torch.zeros(gameAuthority.MoveTensorShape())
        chosenMoveTensor[chosenCoordinates] = 1.0
        return chosenMoveTensor

    def NormalizedMoveProbabilities(self, positionTensor, player, gameAuthority, preApplySoftMax=True, softMaxTemperature=1.0):
        rawMoveProbabilitiesTensor, value = self.forward(positionTensor.unsqueeze(0))  # Add a dummy minibatch
        # Remove the dummy minibatch
        rawMoveProbabilitiesTensor = torch.squeeze(rawMoveProbabilitiesTensor, 0)

        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
        normalizedProbabilitiesTensor = NormalizeProbabilities(rawMoveProbabilitiesTensor,
                                                               legalMovesMask,
                                                               preApplySoftMax=preApplySoftMax,
                                                               softMaxTemperature=softMaxTemperature)
        return normalizedProbabilitiesTensor



def NormalizeProbabilities(moveProbabilitiesTensor, legalMovesMask, preApplySoftMax=True, softMaxTemperature=1.0):
    if moveProbabilitiesTensor.shape != legalMovesMask.shape:
        raise ValueError("NormalizeProbabilities(): The shape of moveProbabilitiesTensor ({}) doesn't match the shape of legalMovesMask ({})".format(moveProbabilitiesTensor, legalMovesMask))
    # Make a copy to avoid changing moveProbabilitiesTensor
    moveProbabilitiesCopyTensor = torch.zeros(moveProbabilitiesTensor.shape)
    moveProbabilitiesCopyTensor.copy_(moveProbabilitiesTensor)
    # Flatten the tensors
    moveProbabilitiesVector = moveProbabilitiesCopyTensor.view(moveProbabilitiesCopyTensor.numel())
    legalMovesVector = legalMovesMask.view(legalMovesMask.numel())
    legalProbabilitiesValues = []

    for index in range(moveProbabilitiesVector.shape[0]):
        if legalMovesVector[index] == 1:
            legalProbabilitiesValues.append(moveProbabilitiesVector[index])

    if preApplySoftMax:
        legalProbabilitiesVector = torch.softmax(torch.Tensor(legalProbabilitiesValues), 0)
        runningNdx = 0
        for index in range(moveProbabilitiesVector.shape[0]):
            if legalMovesVector[index] == 0:
                moveProbabilitiesVector[index] = 0
            else:
                moveProbabilitiesVector[index] = legalProbabilitiesVector[runningNdx]
                runningNdx += 1
    else: # Normalize
        sum = 0
        for index in range(moveProbabilitiesVector.shape[0]):
            if moveProbabilitiesVector[index] < 0:
                raise ValueError("NormalizeProbabilities(): The probability value {} is negative".format(moveProbabilitiesVector[index]))
            sum += moveProbabilitiesVector[index]
        moveProbabilitiesVector = moveProbabilitiesVector/sum

    # Resize to original size
    normalizedProbabilitiesTensor = moveProbabilitiesVector.view(moveProbabilitiesCopyTensor.shape)
    return normalizedProbabilitiesTensor


def ChooseARandomMove(positionTensor, player, gameAuthority):

    legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
    numberOfLegalMoves = torch.nonzero(legalMovesMask).size(0)
    if numberOfLegalMoves == 0:
        raise ValueError("ChooseARandomMove(): The number of legal moves is zero")
    randomNbr = random.randint(1, numberOfLegalMoves)
    probabilitiesTensorShape = legalMovesMask.shape
    runningSum = 0
    chosenCoordinates = None
    for ndx0 in range(probabilitiesTensorShape[0]):
        for ndx1 in range(probabilitiesTensorShape[1]):
            for ndx2 in range(probabilitiesTensorShape[2]):
                for ndx3 in range(probabilitiesTensorShape[3]):
                    runningSum += legalMovesMask[ndx0, ndx1, ndx2, ndx3]
                    if runningSum >= randomNbr and chosenCoordinates is None:
                        chosenCoordinates = (ndx0, ndx1, ndx2, ndx3)

    if chosenCoordinates is None:
        raise IndexError("ChooseARandomMove(): choseCoordinates is None...!???")
    chosenMoveTensor = torch.zeros(gameAuthority.MoveTensorShape())
    chosenMoveTensor[chosenCoordinates] = 1.0
    return chosenMoveTensor

def SimulateGameAndGetReward(playerList, positionTensor, nextPlayer, authority, neuralNetwork,
                             preApplySoftMax, softMaxTemperature):
    winner = None
    if nextPlayer == playerList[0]:
        moveNdx = 0
    elif nextPlayer == playerList[1]:
        moveNdx = 1
    else:
        raise ValueError("SimulateGameAndGetReward(): Unknown player '{}'".format(nextPlayer))
    while winner is None:
        player = playerList[moveNdx % 2]
        chosenMoveTensor = neuralNetwork.ChooseAMove(
            positionTensor,
            player,
            authority,
            preApplySoftMax,
            softMaxTemperature
        )
        #print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
        positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
        moveNdx += 1
        positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
    if winner == playerList[0]:
        return 1.0
    elif winner == 'draw':
        return 0.5
    else:
        return 0

def ProbabilitiesAndValueThroughSelfPlay(playerList,
                                         authority,
                                         neuralNetwork,
                                         startingPositionTensor,
                                         numberOfGamesForEvaluation,
                                         preApplySoftMax,
                                         softMaxTemperature
                                         ):
    legalMovesMask = authority.LegalMovesMask(startingPositionTensor, playerList[0])
    moveTensorShape = authority.MoveTensorShape()
    movesValueTensor = torch.zeros(moveTensorShape)
    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        # print ("nonZeroCoords = \n{}".format(nonZeroCoords))
        firstMoveTensor = torch.zeros(moveTensorShape)
        firstMoveTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        positionAfterFirstMoveTensor, winner = authority.Move(startingPositionTensor, playerList[0],
                                                              firstMoveTensor)
        if winner == playerList[0]:
            averageReward = 1.0
        elif winner == playerList[1]:
            averageReward = 0.0
        elif winner == 'draw':
            averageReward = 0.5
        else:
            rewardSum = 0
            for evaluationGameNdx in range(numberOfGamesForEvaluation):
                reward = SimulateGameAndGetReward(playerList, positionAfterFirstMoveTensor,
                                                  playerList[1], authority, neuralNetwork,
                                                  preApplySoftMax=True, # Helps favor diversity of play while learning
                                                  softMaxTemperature=1.0)
                rewardSum += reward
            averageReward = rewardSum / numberOfGamesForEvaluation
        # print ("averageReward = {}".format(averageReward))
        # Set the value of each move
        movesValueTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = averageReward
    # Calculate the initial position value: weighted average of the value moves
    movesProbabilitiesTensor = NormalizeProbabilities(movesValueTensor,
                                                             legalMovesMask,
                                                             preApplySoftMax=preApplySoftMax,
                                                             softMaxTemperature=softMaxTemperature)
    valueWeightedSum = 0
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        moveValue = movesValueTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
        moveProbability = movesProbabilitiesTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
        valueWeightedSum += moveProbability * moveValue

    return (movesProbabilitiesTensor, valueWeightedSum)

def GeneratePositionToMoveProbabilityAndValueDic(playerList,
                                                 authority,
                                                 neuralNetwork,
                                                 proportionOfRandomInitialPositions,
                                                 maximumNumberOfMovesForInitialPositions,
                                                 numberOfInitialPositions,
                                                 numberOfGamesForEvaluation,
                                                 ):
    # Create initial positions
    initialPositions = []
    randomInitialPositionsNumber = int(proportionOfRandomInitialPositions * numberOfInitialPositions)
    createdRandomInitialPositionsNumber = 0
    while createdRandomInitialPositionsNumber < randomInitialPositionsNumber:
        numberOfMoves = random.randint(0, maximumNumberOfMovesForInitialPositions)
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            player = playerList[moveNdx % 2]
            randomMoveTensor = ChooseARandomMove(positionTensor, player, authority)
            positionTensor, winner = authority.Move(positionTensor, player, randomMoveTensor)
            moveNdx += 1
        if winner is None:
            initialPositions.append(positionTensor)
            createdRandomInitialPositionsNumber += 1

    # Complete with initial positions obtained through self-play
    while len(initialPositions) < numberOfInitialPositions:
        numberOfMoves = random.randint(0, maximumNumberOfMovesForInitialPositions)
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            player = playerList[moveNdx % 2]
            chosenMoveTensor = neuralNetwork.ChooseAMove(positionTensor, player, authority,
                                                         preApplySoftMax=True, softMaxTemperature=1.0)
            positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
            moveNdx += 1
            positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
        if winner is None:
            initialPositions.append(positionTensor)

    # For each initial position, evaluate the value of each possible move through self-play
    positionToMoveProbabilitiesAndValueDic = {}
    for initialPosition in initialPositions:
        positionToMoveProbabilitiesAndValueDic[initialPosition] = \
            ProbabilitiesAndValueThroughSelfPlay(playerList,
                                                 authority,
                                                 neuralNetwork,
                                                 initialPosition,
                                                 numberOfGamesForEvaluation,
                                                 preApplySoftMax=False, # The values are non-negative: don't apply softmax
                                                 softMaxTemperature=0)

    return positionToMoveProbabilitiesAndValueDic

def main():
    print ("policy.py main()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodyStructure', help="The structure of the neural network body. Default: '[(3, 32), (3, 32)]'", default='[(3, 32), (3, 32)]')
    args = parser.parse_args()

    inputTensorSize = (2, 1, 3, 3) # Tic-tac-toe positions (C, D, H, W)
    outputTensorSize = (1, 1, 3, 3) # Tic-tac-toe moves (C, D, H, W)
    neuralNet = NeuralNetwork(inputTensorSize, args.bodyStructure, outputTensorSize)
    input = torch.zeros(inputTensorSize).unsqueeze(0) # Add a dummy mini-batch
    input[0, 0, 0, 0, 0] = 1.0
    output = neuralNet(input)
    print ("main(): output = {}".format(output))
    print ("main(): output[0].shape = {}".format(output[0].shape))


if __name__ == '__main__':
    main()