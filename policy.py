import argparse
import ast
import torch
import math
import random
import sys
import statistics
import numpy


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
        elif ast.literal_eval(bodyStructure) == [(3, 16), (3, 16), (3, 16)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=inputTensorSize[0], out_channels=16,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=16,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=16,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.lastLayerInputNumberOfChannels = 16
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
        elif ast.literal_eval(bodyStructure) == [(7, 1, 1, 16), (7, 1, 1, 16), (7, 1, 1, 16)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=inputTensorSize[0], out_channels=16,
                                kernel_size=(7, 1, 1),
                                padding=(3, 0, 0)),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=16,
                                kernel_size=(7, 1, 1),
                                padding=(3, 0, 0)),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=16,
                                kernel_size=(7, 1, 1),
                                padding=(3, 0, 0)),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.lastLayerInputNumberOfChannels = 16
        elif ast.literal_eval(bodyStructure) == [(15, 1, 1, 16), (15, 1, 1, 16), (15, 1, 1, 16)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=inputTensorSize[0], out_channels=16,
                                kernel_size=(15, 1, 1),
                                padding=(7, 0, 0)),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=16,
                                kernel_size=(15, 1, 1),
                                padding=(7, 0, 0)),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=16, out_channels=16,
                                kernel_size=(15, 1, 1),
                                padding=(7, 0, 0)),
                torch.nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.lastLayerInputNumberOfChannels = 16
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

        #chosenMoveTensor = torch.zeros(gameAuthority.MoveTensorShape())
        chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
        chosenMoveArr[chosenCoordinates] = 1.0
        return torch.from_numpy(chosenMoveArr).float()

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
    #moveProbabilitiesCopyArr = numpy.zeros(moveProbabilitiesTensor.shape)
    moveProbabilitiesCopyTensor.copy_(moveProbabilitiesTensor)
    # Flatten the tensors
    moveProbabilitiesVector = moveProbabilitiesCopyTensor.view(moveProbabilitiesCopyTensor.numel())
    legalMovesVector = legalMovesMask.view(legalMovesMask.numel())
    legalProbabilitiesValues = []

    for index in range(moveProbabilitiesVector.shape[0]):
        if legalMovesVector[index] == 1:
            legalProbabilitiesValues.append(moveProbabilitiesVector[index])

    if preApplySoftMax:
        legalProbabilitiesVector = torch.softmax(torch.Tensor(legalProbabilitiesValues)/softMaxTemperature, 0)
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
    #chosenMoveTensor = torch.zeros(gameAuthority.MoveTensorShape())
    chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
    chosenMoveArr[chosenCoordinates] = 1.0
    return torch.from_numpy(chosenMoveArr).float()

def SimulateGameAndGetReward(playerList,
                             positionTensor,
                             nextPlayer,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax,
                             softMaxTemperature):
    winner = None
    if nextPlayer == playerList[0]:
        moveNdx = 0
    elif nextPlayer == playerList[1]:
        moveNdx = 1
    else:
        raise ValueError("SimulateGameAndGetReward(): Unknown player '{}'".format(nextPlayer))
    while winner is None:
        player = playerList[moveNdx % 2]
        if neuralNetwork is None:
            chosenMoveTensor = ChooseARandomMove(positionTensor, player, authority)
        else:
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
        return 0.0
    else:
        return -1.0

def SimulateGameAndGetPositionsMovesListReward(playerList,
                             positionTensor,
                             nextPlayer,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax,
                             softMaxTemperature):
    #print ("SimulateGameAndGetPositionsMovesListReward(): \npositionTensor = \n{}".format(positionTensor))
    player0PositionMovesList = list()
    player1PositionMovesList = list()
    winner = None
    #authority.Display(positionTensor)
    if nextPlayer == playerList[0]:
        moveNdx = 0
    elif nextPlayer == playerList[1]:
        moveNdx = 1
    else:
        raise ValueError("SimulateGameAndGetPositionsMovesListReward(): Unknown player '{}'".format(nextPlayer))
    if nextPlayer == playerList[1]:
        positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
    while winner is None:
        player = playerList[moveNdx % 2]

        if neuralNetwork is None:
            chosenMoveTensor = ChooseARandomMove(positionTensor, playerList[0], authority)
        else:
            chosenMoveTensor = neuralNetwork.ChooseAMove(
                positionTensor,
                playerList[0],
                authority,
                preApplySoftMax,
                softMaxTemperature
            )
        #print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))

        positionBeforeMoveTensor = positionTensor.clone()
        #print ("SimulateGameAndGetPositionsMovesListReward(): positionBeforeMoveTensor = {}".format(positionBeforeMoveTensor))
        positionTensor, winner = authority.Move(positionTensor, playerList[0], chosenMoveTensor)
        #authority.Display(positionTensor)
        if player == playerList[0]:
            player0PositionMovesList.append( (positionBeforeMoveTensor, chosenMoveTensor) )
        else:
            player1PositionMovesList.append((positionBeforeMoveTensor, chosenMoveTensor))
        if winner == playerList[0] and player == playerList[1]: # All moves are from the point of view of player0, hence he will always 'win'
            winner = playerList[1]
        moveNdx += 1
        positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
    if winner == playerList[0]:
        return (player0PositionMovesList, 1.0), (player1PositionMovesList, -1.0)
    elif winner == 'draw':
        return (player0PositionMovesList, 0.0), (player1PositionMovesList, 0.0)
    else:
        return (player0PositionMovesList, -1.0), (player1PositionMovesList, 1.0)

def ProbabilitiesAndValueThroughSelfPlay(playerList,
                                         authority,
                                         neuralNetwork, # If None, do random moves
                                         startingPositionTensor,
                                         numberOfGamesForEvaluation,
                                         preApplySoftMax,
                                         softMaxTemperature,
                                         numberOfStandardDeviationsBelowAverageForValueEstimate
                                         ):
    legalMovesMask = authority.LegalMovesMask(startingPositionTensor, playerList[0])
    moveTensorShape = authority.MoveTensorShape()
    #movesValueTensor = torch.zeros(moveTensorShape)
    movesValueArr = numpy.zeros(moveTensorShape)
    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        # print ("nonZeroCoords = \n{}".format(nonZeroCoords))
        firstMoveTensor = torch.zeros(moveTensorShape)
        firstMoveTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        positionAfterFirstMoveTensor, winner = authority.Move(startingPositionTensor, playerList[0],
                                                              firstMoveTensor)
        if winner == playerList[0]:
            #averageReward = 1.0
            valueEstimate = 1.0
        elif winner == playerList[1]:
            #averageReward = -1.0
            valueEstimate = -1.0
        elif winner == 'draw':
            #averageReward = 0.0
            valueEstimate = 0.0
        else:
            #rewardSum = 0
            #minimumReward = sys.float_info.max
            rewardsList = []
            for evaluationGameNdx in range(numberOfGamesForEvaluation):
                reward = SimulateGameAndGetReward(playerList, positionAfterFirstMoveTensor,
                                                  playerList[1], authority, neuralNetwork,
                                                  preApplySoftMax=True, # Helps favor diversity of play while learning
                                                  softMaxTemperature=1.0)
                #rewardSum += reward
                #if reward < minimumReward:
                #    minimumReward = reward
                rewardsList.append(reward)
            #averageReward = rewardSum / numberOfGamesForEvaluation
            # print ("averageReward = {}".format(averageReward))
            averageReward = statistics.mean(rewardsList)
            rewardStdDev = statistics.stdev(rewardsList)
            valueEstimate = averageReward - numberOfStandardDeviationsBelowAverageForValueEstimate * rewardStdDev
        # Set the value of each move
        movesValueArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = valueEstimate
    # Calculate the initial position value: weighted average of the value moves
    movesValueTensor = torch.from_numpy(movesValueArr).float()
    movesProbabilitiesTensor = NormalizeProbabilities(movesValueTensor,
                                                             legalMovesMask,
                                                             preApplySoftMax=preApplySoftMax,
                                                             softMaxTemperature=softMaxTemperature)
    """valueWeightedSum = 0
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        moveValue = movesValueTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
        moveProbability = movesProbabilitiesTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
        valueWeightedSum += moveProbability * moveValue
    """
    maxValue = sys.float_info.min
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        moveValue = movesValueTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
        if moveValue > maxValue:
            maxValue = moveValue

    return (movesProbabilitiesTensor, maxValue)

def IntermediateStatesProbabilitiesAndValue(
                playerList,
                authority,
                neuralNetwork,
                startingPositionTensor,
                numberOfGamesForEvaluation,
                softMaxTemperatureForSelfPlayEvaluation
            ):
    legalMovesMask = authority.LegalMovesMask(startingPositionTensor, playerList[0])
    moveTensorShape = authority.MoveTensorShape()

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    firstMoveProbabilities, startingPositionValue = neuralNetwork(startingPositionTensor.unsqueeze(0))
    firstMoveProbabilities = firstMoveProbabilities.squeeze(0).detach() # Remove the dummy mini-batch
    intermediateStateProbabilitiesAndValues = list()
    #print ("IntermediateStatesProbabilitiesAndValue(): nonZeroCoordsTensor.shape = {}".format(nonZeroCoordsTensor.shape))
    #print ("IntermediateStatesProbabilitiesAndValue(): nonZeroCoordsTensor = {}".format(nonZeroCoordsTensor))
    #print ("IntermediateStatesProbabilitiesAndValue(): nonZeroCoordsTensor.size(0) = {}".format(nonZeroCoordsTensor.size(0)))
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        #print ("IntermediateStatesToProbabilitiesAndValue(): nonZeroCoords = {}".format(nonZeroCoords))
        firstMoveArr = numpy.zeros(moveTensorShape)
        firstMoveArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        firstMoveTensor = torch.from_numpy(firstMoveArr).float()
        positionAfterFirstMoveTensor, winner = authority.Move(startingPositionTensor, playerList[0],
                                                              firstMoveTensor)
        if winner == playerList[0]:
            probabilitiesArr = numpy.zeros(moveTensorShape) - 1.0/NumberOfEntries(moveTensorShape)
            probabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = \
                probabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] + 1.0 # Good move!
            correctedProbabilitiesTensor = firstMoveProbabilities + torch.from_numpy(probabilitiesArr).float()

            intermediateStateProbabilitiesAndValues.append((startingPositionTensor.clone(), correctedProbabilitiesTensor, 1.0))

        elif winner == playerList[1]:
            probabilitiesArr = numpy.zeros(moveTensorShape) + 1.0/NumberOfEntries(moveTensorShape)
            probabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = \
                probabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] - 1.0  # Bad move!
            correctedProbabilitiesTensor = firstMoveProbabilities + torch.from_numpy(probabilitiesArr).float()

            intermediateStateProbabilitiesAndValues.append( (startingPositionTensor.clone(), \
                correctedProbabilitiesTensor, -1.0) )

        elif winner == 'draw':
            # Draw game: Don't record anything (it would be zeros everywhere in the probability tensor)
            #probabilitiesArr = numpy.zeros(moveTensorShape)
            #probabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            """intermediateStateToProbabilitiesAndValueDic[positionAfterFirstMoveTensor] = (
                firstMoveProbabilities, 0.0
            )
            """
            intermediateStateProbabilitiesAndValues.append( (startingPositionTensor.clone(), \
                                                             firstMoveProbabilities, 0.0) )

        else: # The game is not over: Let's simulate the evolution of the game
            for evaluationGameNdx in range(numberOfGamesForEvaluation):
                (player0PositionsMovesList, player0Reward), (player1PositionsMovesList, player1Reward) = \
                    SimulateGameAndGetPositionsMovesListReward(
                        playerList, positionAfterFirstMoveTensor,
                        playerList[1], authority, neuralNetwork,
                        preApplySoftMax=True,  # Helps favor diversity of play while learning
                        softMaxTemperature=softMaxTemperatureForSelfPlayEvaluation
                    )
                player0PositionsMovesList.append((startingPositionTensor.clone(), firstMoveTensor.clone()))
                #if reward == 1.: # The current player won; the opponent lost
                player0CorrectionProbabilitiesArr = numpy.zeros(moveTensorShape) -1.0 * player0Reward / NumberOfEntries(moveTensorShape)
                player0CorrectionProbabilitiesArr[
                    nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = \
                    player0CorrectionProbabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] + player0Reward
                player0CorrectionProbabilitiesTensor = torch.from_numpy(player0CorrectionProbabilitiesArr).float()
                """correctedProbabilityTensor = aPrioriProbabilityTensor + 
                intermediateStateToProbabilitiesAndValueDic[positionAfterFirstMoveTensor] = (
                    torch.from_numpy(player0ProbabilitiesArr).float(), player0Reward
                )
                intermediateStateToProbabilitiesAndValueDic[startingPositionTensor] = (
                    firstMoveTensor, player0Reward
                )"""
                for positionNdx in range(len(player0PositionsMovesList)):
                    position = player0PositionsMovesList[positionNdx][0]
                    aPrioriProbabilityTensor, aPrioriValue = neuralNetwork(position.unsqueeze(0))
                    aPrioriProbabilityTensor = aPrioriProbabilityTensor.squeeze(0).detach() # Remove the dummy mini-batch
                    chosenMove = player0PositionsMovesList[positionNdx][1]
                    #sign = (-1.) ** (positionNdx + 1)
                    correctedMoveProbability = aPrioriProbabilityTensor + player0CorrectionProbabilitiesTensor + \
                        chosenMove * player0Reward

                    intermediateStateProbabilitiesAndValues.append( (position.clone(), correctedMoveProbability, player0Reward) )
                    #print ("IntermediateStatesProbabilitiesAndValue(): position = {}; correctedMoveProbability = {}; player0Reward = {}".format(position, correctedMoveProbability, player0Reward))
                #elif reward == -1: # The current player lost; the opponent won
                player1CorrectionProbabilitiesArr = numpy.zeros(moveTensorShape) - 1.0 * player1Reward / NumberOfEntries(moveTensorShape)
                player1CorrectionProbabilitiesArr[
                    nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = \
                    player1CorrectionProbabilitiesArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] + player1Reward
                player1CorrectionProbabilitiesTensor = torch.from_numpy(player1CorrectionProbabilitiesArr).float()
                """intermediateStateToProbabilitiesAndValueDic[positionAfterFirstMoveTensor] = (
                    torch.from_numpy(player1ProbabilitiesArr).float(), player1Reward
                )
                intermediateStateToProbabilitiesAndValueDic[startingPositionTensor] = (
                    player1Reward * firstMoveTensor, player1Reward
                )"""
                for positionNdx in range(len(player1PositionsMovesList)):
                    position = player1PositionsMovesList[positionNdx][0]
                    aPrioriProbabilityTensor, aPrioriValue = neuralNetwork(position.unsqueeze(0))
                    aPrioriProbabilityTensor = aPrioriProbabilityTensor.squeeze(0).detach()  # Remove the dummy mini-batch
                    chosenMove = player1PositionsMovesList[positionNdx][1]
                    #sign = (-1.) ** (positionNdx)
                    correctedMoveProbability = aPrioriProbabilityTensor + player1CorrectionProbabilitiesTensor + \
                                               chosenMove * player1Reward

                    intermediateStateProbabilitiesAndValues.append( (position.clone(), correctedMoveProbability, player1Reward) )
                    #print ("IntermediateStatesProbabilitiesAndValue(): position = {}; correctedMoveProbability = {}; player1Reward = {}".format(
                    #        position, correctedMoveProbability, player1Reward))

    return intermediateStateProbabilitiesAndValues


def GeneratePositionMoveProbabilityAndValue(playerList,
                                                 authority,
                                                 neuralNetwork,
                                                 proportionOfRandomInitialPositions,
                                                 maximumNumberOfMovesForInitialPositions,
                                                 numberOfInitialPositions,
                                                 numberOfGamesForEvaluation,
                                                 #numberOfStandardDeviationsBelowAverageForValueEstimate,
                                                 softMaxTemperatureForSelfPlayEvaluation
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
            initialPositions.append(positionTensor.clone())
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
            initialPositions.append(positionTensor.clone())

    # For each initial position, evaluate the value of each possible move through self-play
    positionMoveProbabilitiesAndValues = list()
    for initialPosition in initialPositions:
        """positionToMoveProbabilitiesAndValueDic[initialPosition] = \
            ProbabilitiesAndValueThroughSelfPlay(playerList,
                                                 authority,
                                                 neuralNetwork,
                                                 initialPosition,
                                                 numberOfGamesForEvaluation,
                                                 True,
                                                 softMaxTemperatureForSelfPlayEvaluation,
                                                 numberOfStandardDeviationsBelowAverageForValueEstimate)
        """
        intermediateStatesProbabilitiesAndValues = \
            IntermediateStatesProbabilitiesAndValue(
                playerList,
                authority,
                neuralNetwork,
                initialPosition,
                numberOfGamesForEvaluation,
                softMaxTemperatureForSelfPlayEvaluation
            )
        for (intermediateState, probabilities, value) in intermediateStatesProbabilitiesAndValues:
            positionMoveProbabilitiesAndValues.append( (intermediateState, probabilities, value) )

    return positionMoveProbabilitiesAndValues


def MinibatchIndices(numberOfSamples, minibatchSize):
	shuffledList = numpy.arange(numberOfSamples)
	numpy.random.shuffle(shuffledList)
	minibatchesIndicesList = []
	numberOfWholeLists = int(numberOfSamples / minibatchSize)
	for wholeListNdx in range(numberOfWholeLists):
		minibatchIndices = shuffledList[ wholeListNdx * minibatchSize : (wholeListNdx + 1) * minibatchSize ]
		minibatchesIndicesList.append(minibatchIndices)
	# Add the last incomplete minibatch
	if numberOfWholeLists * minibatchSize < numberOfSamples:
		lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
		minibatchesIndicesList.append(lastMinibatchIndices)
	return minibatchesIndicesList

def MinibatchTensor(positionsList):
    if len(positionsList) == 0:
        raise ArgumentException("MinibatchTensor(): Empty list of positions")
    #print ("MinibatchTensor(): len(positionsList) = {}; positionsList[0].shape = {}".format(len(positionsList), positionsList[0].shape) )
    positionShape = positionsList[0].shape
    minibatchTensor = torch.zeros(len(positionsList), positionShape[0],
                                  positionShape[1], positionShape[2], positionShape[3]) # NCDHW
    for n in range(len(positionsList)):
        #print ("MinibatchTensor(): positionsList[n].shape = {}".format(positionsList[n].shape))
        #print ("MinibatchTensor(): positionsList[n] = {}".format(positionsList[n]))
        minibatchTensor[n] = positionsList[n]
    return minibatchTensor

def MinibatchValuesTensor(valuesList):
    if len(valuesList) == 0:
        raise ArgumentException("MinibatchValuesTensor(): Empty list of values")
    valuesTensor = torch.zeros(len (valuesList))
    for n in range (len (valuesList)):
        valuesTensor[n] = valuesList[n]
    return valuesTensor

def adjust_lr(optimizer, desiredLearningRate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = desiredLearningRate

def AverageRewardAgainstARandomPlayer(
                             playerList,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax,
                             softMaxTemperature,
                             numberOfGames):
    rewardSum = 0
    numberOfWins = 0
    numberOfDraws = 0
    numberOfLosses = 0
    for gameNdx in range(numberOfGames):
        firstPlayer = playerList[gameNdx % 2]
        if firstPlayer == playerList[0]:
            moveNdx = 0
        else:
            moveNdx = 1
        positionTensor = authority.InitialPosition()
        #authority.Display(positionTensor)
        winner = None
        while winner is None:
            player = playerList[moveNdx % 2]
            if player == playerList[1] or neuralNetwork is None:
                chosenMoveTensor = ChooseARandomMove(positionTensor, player, authority)
            else:
                chosenMoveTensor = neuralNetwork.ChooseAMove(
                    positionTensor,
                    player,
                    authority,
                    preApplySoftMax,
                    softMaxTemperature
                )
            # print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
            positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
            #authority.Display(positionTensor)
            #print ("policy.AverageRewardAgainstARandomPlayer() winner = {}".format(winner))
            moveNdx += 1
            #positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
        if winner == playerList[0]:
            rewardSum += 1.0
            numberOfWins += 1
        elif winner == 'draw':
            rewardSum += 0.0
            numberOfDraws += 1
        else:
            rewardSum += -1.0
            numberOfLosses += 1

    return (rewardSum / numberOfGames, numberOfWins / numberOfGames, numberOfDraws / numberOfGames,
        numberOfLosses / numberOfGames)

def NumberOfEntries(tensorShape):
    return tensorShape[0] * tensorShape[1] * tensorShape[2] * tensorShape[3]

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