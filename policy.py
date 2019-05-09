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

        self.actionValuesChannelMatcher = torch.nn.Conv3d(in_channels=self.lastLayerInputNumberOfChannels,
                                              out_channels=outputTensorSize[0],
                                              kernel_size=1,
                                              padding=0
                                              )
        self.actionValuesResizer = torch.nn.Upsample(size=outputTensorSize[-3:], mode='trilinear')
        self.outputTensorSize = outputTensorSize
        self.lastLayerNumberOfFeatures = self.lastLayerInputNumberOfChannels * \
            inputTensorSize[-3] * inputTensorSize[-2] * inputTensorSize [-1]

    def forward(self, inputs):
        # Compute the output of the body
        bodyOutputTensor = self.bodyStructure(inputs)

        # Move probabilities
        actionValuesActivation = self.actionValuesChannelMatcher(bodyOutputTensor)
        actionValuesTensor = self.actionValuesResizer(actionValuesActivation)
        return actionValuesTensor

    def ChooseAMove(self, positionTensor, player, gameAuthority, preApplySoftMax=True, softMaxTemperature=1.0,
                    epsilon=0.1):
        actionValuesTensor = self.forward(positionTensor.unsqueeze(0)) # Add a dummy minibatch
        # Remove the dummy minibatch
        actionValuesTensor = torch.squeeze(actionValuesTensor, 0)

        chooseARandomMove = random.random() < epsilon
        if chooseARandomMove:
            return ChooseARandomMove(positionTensor, player, gameAuthority)

        # Else: choose according to probabilities
        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)

        normalizedActionValuesTensor = NormalizeProbabilities(actionValuesTensor,
                                                               legalMovesMask,
                                                               preApplySoftMax=preApplySoftMax,
                                                               softMaxTemperature=softMaxTemperature)
        randomNbr = random.random()
        actionValuesTensorShape = normalizedActionValuesTensor.shape

        runningSum = 0
        chosenCoordinates = None
        for ndx0 in range(actionValuesTensorShape[0]):
            for ndx1 in range(actionValuesTensorShape[1]):
                for ndx2 in range(actionValuesTensorShape[2]):
                    for ndx3 in range(actionValuesTensorShape[3]):
                        runningSum += normalizedActionValuesTensor[ndx0, ndx1, ndx2, ndx3]
                        if runningSum >= randomNbr and chosenCoordinates is None:
                            chosenCoordinates = (ndx0, ndx1, ndx2, ndx3)

        if chosenCoordinates is None:
            print ("NeuralNetwork.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
            print ("NeuralNetwork.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
            print ("NeuralNetwork.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
            print ("NeuralNetwork.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
            raise IndexError("NeuralNetwork.ChooseAMove(): chosenCoordinates is None...!???")

        chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
        chosenMoveArr[chosenCoordinates] = 1.0
        return torch.from_numpy(chosenMoveArr).float()

    def HighestActionValueMove(self, positionTensor, player, gameAuthority):
        actionValuesTensor = self.forward(positionTensor.unsqueeze(0))  # Add a dummy minibatch
        # Remove the dummy minibatch
        actionValuesTensor = torch.squeeze(actionValuesTensor, 0)
        chosenMoveTensor = torch.zeros(gameAuthority.MoveTensorShape())
        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
        highestValue = -1E9
        highestValueCoords = (0, 0, 0, 0)
        nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
        for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
            nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
            #print ("HighestProbabilityMove(): nonZeroCoords = {}".format(nonZeroCoords))
            if actionValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] > highestValue:
                highestValue = actionValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
                highestValueCoords = nonZeroCoords
        chosenMoveTensor[ highestValueCoords[0], highestValueCoords[1], highestValueCoords[2], highestValueCoords[3] ] = 1.0
        return chosenMoveTensor


    def NormalizedMoveProbabilities(self, positionTensor, player, gameAuthority, preApplySoftMax=True, softMaxTemperature=1.0):
        actionValuesTensor = self.forward(positionTensor.unsqueeze(0))  # Add a dummy minibatch
        # Remove the dummy minibatch
        actionValuesTensor = torch.squeeze(actionValuesTensor, 0)

        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
        normalizedActionValuesTensor = NormalizeProbabilities(actionValuesTensor,
                                                               legalMovesMask,
                                                               preApplySoftMax=preApplySoftMax,
                                                               softMaxTemperature=softMaxTemperature)
        return normalizedActionValuesTensor



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

    chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
    chosenMoveArr[chosenCoordinates] = 1.0
    return torch.from_numpy(chosenMoveArr).float()

def SimulateGameAndGetReward(playerList,
                             positionTensor,
                             nextPlayer,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax,
                             softMaxTemperature,
                             epsilon):
    winner = None
    if nextPlayer == playerList[0]:
        moveNdx = 0
    elif nextPlayer == playerList[1]:
        moveNdx = 1
    else:
        raise ValueError("SimulateGameAndGetReward(): Unknown player '{}'".format(nextPlayer))
    if nextPlayer == playerList[1]:
        positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
    while winner is None:
        #print ("SimulateGameAndGetReward(): positionTensor = {}".format(positionTensor))
        player = playerList[moveNdx % 2]
        if neuralNetwork is None:
            chosenMoveTensor = ChooseARandomMove(positionTensor, playerList[0], authority)
        else:
            chosenMoveTensor = neuralNetwork.ChooseAMove(
                positionTensor,
                playerList[0],
                authority,
                preApplySoftMax,
                softMaxTemperature,
                epsilon=epsilon
            ).detach()
        #print ("SimulateGameAndGetReward(): chosenMoveTensor =\n{}".format(chosenMoveTensor))
        positionTensor, winner = authority.Move(positionTensor, playerList[0], chosenMoveTensor)
        if winner == playerList[0] and player == playerList[1]: # All moves are from the point of view of player0, hence he will always 'win'
            winner = playerList[1]
        moveNdx += 1
        #print ("SimulateGameAndGetReward(): After move: positionTensor = {}".format(positionTensor))
        positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
        #print ("SimulateGameAndGetReward(): After swap: positionTensor = {}".format(positionTensor))
    if winner == playerList[0]:
        return 1.0
    elif winner == 'draw':
        return 0.0
    else:
        return -1.0

"""
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
            ).detach()
        #print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))

        positionBeforeMoveTensor = positionTensor.clone().detach()
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
"""

"""
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

    maxValue = sys.float_info.min
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        moveValue = movesValueTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
        if moveValue > maxValue:
            maxValue = moveValue

    return (movesProbabilitiesTensor, maxValue)
"""

"""
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
"""

"""
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
"""

"""
def PositionExpectedMoveValues(
        playerList,
        authority,
        neuralNetwork,
        initialPosition,
        numberOfGamesForEvaluation,
        softMaxTemperatureForSelfPlayEvaluation,
        epsilon
        ):
    legalMovesMask = authority.LegalMovesMask(initialPosition, playerList[0])
    moveTensorShape = authority.MoveTensorShape()

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    moveValuesTensor = torch.zeros(moveTensorShape)
    standardDeviationTensor = torch.zeros(moveTensorShape) - 1.0
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        firstMoveArr = numpy.zeros(moveTensorShape)
        firstMoveArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        firstMoveTensor = torch.from_numpy(firstMoveArr).float()
        positionAfterFirstMoveTensor, winner = authority.Move(initialPosition, playerList[0],
                                                              firstMoveTensor)
        if winner == playerList[0]:
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ] = 1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ] = 0.0
        elif winner == 'draw':
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
        elif winner == playerList[1]: # The opponent won
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = -1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
        else: # winner == None
            rewards = []
            for simulationNdx in range(numberOfGamesForEvaluation):
                reward = SimulateGameAndGetReward(
                    playerList,
                    positionAfterFirstMoveTensor,
                    playerList[1],
                    authority,
                    neuralNetwork,
                    preApplySoftMax=True,
                    softMaxTemperature=softMaxTemperatureForSelfPlayEvaluation,
                    epsilon=epsilon
                )
                rewards.append(reward)
            averageReward = statistics.mean(rewards)
            if len(rewards) > 1:
                standardDeviation = statistics.stdev(rewards)
            else:
                standardDeviation = 0
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = averageReward
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = standardDeviation
    return moveValuesTensor, standardDeviationTensor, legalMovesMask
"""

def PositionExpectedMoveValues(
        playerList,
        authority,
        neuralNetwork,
        initialPosition,
        numberOfGamesForEvaluation,
        softMaxTemperatureForSelfPlayEvaluation,
        epsilon,
        depthOfExhaustiveSearch
        ):
    legalMovesMask = authority.LegalMovesMask(initialPosition, playerList[0])
    moveTensorShape = authority.MoveTensorShape()

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    moveValuesTensor = torch.zeros(moveTensorShape)
    standardDeviationTensor = torch.zeros(moveTensorShape) - 1.0
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        firstMoveArr = numpy.zeros(moveTensorShape)
        firstMoveArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        firstMoveTensor = torch.from_numpy(firstMoveArr).float()
        positionAfterFirstMoveTensor, winner = authority.Move(initialPosition, playerList[0],
                                                              firstMoveTensor)
        #print ("PositionExpectedMoveValues2(): positionAfterFirstMoveTensor = {}".format(positionAfterFirstMoveTensor))
        if winner == playerList[0]:
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ] = 1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ] = 0.0
        elif winner == 'draw':
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
        elif winner == playerList[1]: # The opponent won
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = -1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
        else: # winner == None
            #print ("PositionExpectedMoveValues2(): Go with RewardStatistics()")
            (rewardAverage, rewardStandardDeviation) = RewardStatistics(
                positionAfterFirstMoveTensor,
                2,
                depthOfExhaustiveSearch,
                playerList,
                playerList[1],
                authority,
                neuralNetwork,
                softMaxTemperatureForSelfPlayEvaluation,
                epsilon,
                numberOfGamesForEvaluation
            )
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardAverage
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardStandardDeviation
        #print ("PositionExpectedMoveValues2(): high-level moveValuesTensor = {}".format(moveValuesTensor))

    return moveValuesTensor, standardDeviationTensor, legalMovesMask

def RewardStatistics(positionTensor, searchDepth, maxSearchDepth, playersList, player, authority,
                  neuralNetwork, softMaxTemperatureForSelfPlayEvaluation, epsilon,
                  numberOfGamesForEvaluation):
    if searchDepth > maxSearchDepth: # Evaluate with Monte-Carlo tree search
        rewards = []
        for simulationNdx in range(numberOfGamesForEvaluation):
            reward = SimulateGameAndGetReward(
                playersList,
                positionTensor,
                player,
                authority,
                neuralNetwork,
                preApplySoftMax=True,
                softMaxTemperature=softMaxTemperatureForSelfPlayEvaluation,
                epsilon=epsilon
            )
            rewards.append(reward)
        averageReward = statistics.mean(rewards)
        if len(rewards) > 1:
            standardDeviation = statistics.stdev(rewards)
        else:
            standardDeviation = 0
        return (averageReward, standardDeviation)
    else: # searchDepth <= maxSearchDepth => Exhaustive search
        legalMovesMask = authority.LegalMovesMask(positionTensor, player)
        moveTensorShape = authority.MoveTensorShape()
        if player == playersList[0]:
            nextPlayer = playersList[1]
        else:
            nextPlayer = playersList[0]
        nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
        moveValuesTensor = torch.zeros(moveTensorShape)
        standardDeviationTensor = torch.zeros(moveTensorShape) - 1.0
        #print ("RewardStatistics(): {}".format(player))
        #print ("RewardStatistics(): {}".format(positionTensor))
        for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
            nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
            moveArr = numpy.zeros(moveTensorShape)
            moveArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
            moveTensor = torch.from_numpy(moveArr).float()
            positionAfterMoveTensor, winner = authority.Move(positionTensor, player,
                                                                  moveTensor)
            if winner == playersList[0]:
                rewardAverage = 1.0
                rewardStandardDeviation = 0
            elif winner == 'draw':
                rewardAverage = 0.0
                rewardStandardDeviation = 0
            elif winner == playersList[1]:  # The opponent won
                rewardAverage = -1.0
                rewardStandardDeviation = 0
            else:  # winner == None: Recursive call
                (rewardAverage, rewardStandardDeviation) = RewardStatistics(
                    positionAfterMoveTensor,
                    searchDepth + 1,
                    maxSearchDepth,
                    playersList,
                    nextPlayer,
                    authority,
                    neuralNetwork,
                    softMaxTemperatureForSelfPlayEvaluation,
                    epsilon,
                    numberOfGamesForEvaluation
                )
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardAverage
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardStandardDeviation
        # Return the move statistics for the highest (playersList[0]) or lowest (playersList[1]) average reward

        if player == playersList[0]:
            highestRewardAverage = -1E9
            correspondingStandardDeviation = 0
            for index0 in range(moveTensorShape[0]):
                for index1 in range(moveTensorShape[1]):
                    for index2 in range(moveTensorShape[2]):
                        for index3 in range(moveTensorShape[3]):
                            if legalMovesMask[index0, index1, index2, index3] > 0 and \
                              moveValuesTensor[index0, index1, index2, index3] > highestRewardAverage:
                                highestRewardAverage = moveValuesTensor[index0, index1, index2, index3]
                                correspondingStandardDeviation = standardDeviationTensor[index0, index1, index2, index3]
            #print ("RewardStatistics(): Chose highestRewardAverage = {}".format(highestRewardAverage))
            return (highestRewardAverage, correspondingStandardDeviation)

        else: # playersList[1]: Keep the lowest reward average
            lowestRewardAverage = 1E9
            correspondingStandardDeviation = 0
            for index0 in range(moveTensorShape[0]):
                for index1 in range(moveTensorShape[1]):
                    for index2 in range(moveTensorShape[2]):
                        for index3 in range(moveTensorShape[3]):
                            if legalMovesMask[index0, index1, index2, index3] > 0 and \
                                    moveValuesTensor[index0, index1, index2, index3] < lowestRewardAverage:
                                lowestRewardAverage = moveValuesTensor[index0, index1, index2, index3]
                                correspondingStandardDeviation = standardDeviationTensor[index0, index1, index2, index3]
            #print ("RewardStatistics(): Chose lowestRewardAverage = {}".format(lowestRewardAverage))
            return (lowestRewardAverage, correspondingStandardDeviation)

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

"""
def MinibatchValuesTensor(valuesList):
    if len(valuesList) == 0:
        raise ArgumentException("MinibatchValuesTensor(): Empty list of values")
    valuesTensor = torch.zeros(len (valuesList))
    for n in range (len (valuesList)):
        valuesTensor[n] = valuesList[n]
    return valuesTensor
"""

def adjust_lr(optimizer, desiredLearningRate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = desiredLearningRate

def AverageRewardAgainstARandomPlayer(
                             playerList,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax,
                             softMaxTemperature,
                             numberOfGames,
                             moveChoiceMode='SoftMax',
                             numberOfGamesForMoveEvaluation=31):
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
                if moveChoiceMode == 'SoftMax':
                    chosenMoveTensor = neuralNetwork.ChooseAMove(
                        positionTensor,
                        player,
                        authority,
                        preApplySoftMax,
                        softMaxTemperature,
                        epsilon=0
                    )
                elif moveChoiceMode == 'HighestActionValueMove':
                    chosenMoveTensor = neuralNetwork.HighestActionValueMove(
                        positionTensor, player, authority
                    )
                elif moveChoiceMode == 'ExpectedMoveValuesThroughSelfPlay':
                    moveValuesTensor, standardDeviationTensor, legalMovesMask = PositionExpectedMoveValues(
                        playerList,
                        authority,
                        neuralNetwork,
                        positionTensor,
                        numberOfGamesForMoveEvaluation,
                        softMaxTemperature,
                        epsilon=0
                    )
                    chosenMoveTensor = torch.zeros(authority.MoveTensorShape())
                    highestValue = -1E9
                    highestValueCoords = (0, 0, 0, 0)
                    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
                    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
                        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
                        if moveValuesTensor[
                            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] > highestValue:
                            highestValue = moveValuesTensor[
                                nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
                            highestValueCoords = nonZeroCoords
                    chosenMoveTensor[
                        highestValueCoords[0], highestValueCoords[1], highestValueCoords[2], highestValueCoords[
                            3]] = 1.0
                else:
                    raise NotImplementedError("AverageRewardAgainstARandomPlayer(): Unknown move choice mode '{}'".format(moveChoiceMode))
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

def AverageRewardAgainstARandomPlayerKeepLosingGames(
                             playerList,
                             authority,
                             neuralNetwork, # If None, do random moves
                             preApplySoftMax,
                             softMaxTemperature,
                             numberOfGames,
                             moveChoiceMode='SoftMax',
                             numberOfGamesForMoveEvaluation=31,
                             ):
    rewardSum = 0
    numberOfWins = 0
    numberOfDraws = 0
    numberOfLosses = 0
    losingGamesPositionsListList = list()
    for gameNdx in range(numberOfGames):
        firstPlayer = playerList[gameNdx % 2]
        if firstPlayer == playerList[0]:
            moveNdx = 0
        else:
            moveNdx = 1
        positionTensor = authority.InitialPosition()
        gamePositionsList = list()
        gamePositionsList.append(positionTensor)
        #authority.Display(positionTensor)
        winner = None
        while winner is None:
            player = playerList[moveNdx % 2]
            if player == playerList[1] or neuralNetwork is None:
                chosenMoveTensor = ChooseARandomMove(positionTensor, player, authority)
            else:
                if moveChoiceMode == 'SoftMax':
                    chosenMoveTensor = neuralNetwork.ChooseAMove(
                        positionTensor,
                        player,
                        authority,
                        preApplySoftMax,
                        softMaxTemperature,
                        epsilon=0
                    )
                elif moveChoiceMode == 'HighestActionValueMove':
                    chosenMoveTensor = neuralNetwork.HighestActionValueMove(
                        positionTensor, player, authority
                    )
                elif moveChoiceMode == 'ExpectedMoveValuesThroughSelfPlay':
                    moveValuesTensor, standardDeviationTensor, legalMovesMask = PositionExpectedMoveValues(
                        playerList,
                        authority,
                        neuralNetwork,
                        positionTensor,
                        numberOfGamesForMoveEvaluation,
                        softMaxTemperature,
                        epsilon=0
                    )
                    chosenMoveTensor = torch.zeros(authority.MoveTensorShape())
                    highestValue = -1E9
                    highestValueCoords = (0, 0, 0, 0)
                    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
                    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
                        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
                        if moveValuesTensor[
                            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] > highestValue:
                            highestValue = moveValuesTensor[
                                nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
                            highestValueCoords = nonZeroCoords
                    chosenMoveTensor[
                        highestValueCoords[0], highestValueCoords[1], highestValueCoords[2], highestValueCoords[
                            3]] = 1.0
                else:
                    raise NotImplementedError("AverageRewardAgainstARandomPlayer(): Unknown move choice mode '{}'".format(moveChoiceMode))
            # print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
            positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
            gamePositionsList.append(positionTensor)
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
            losingGamesPositionsListList.append((gamePositionsList, firstPlayer))

    return (rewardSum / numberOfGames, numberOfWins / numberOfGames, numberOfDraws / numberOfGames,
        numberOfLosses / numberOfGames, losingGamesPositionsListList)

"""
def NumberOfEntries(tensorShape):
    return tensorShape[0] * tensorShape[1] * tensorShape[2] * tensorShape[3]
"""

"""
def GenerateMoveStatistics(playerList,
                            authority,
                            neuralNetwork,
                            proportionOfRandomInitialPositions,
                            maximumNumberOfMovesForInitialPositions,
                            numberOfInitialPositions,
                            numberOfGamesForEvaluation,
                            softMaxTemperatureForSelfPlayEvaluation,
                            epsilon,
                            additionalStartingPositionsList=[]
                            ):
    # Create initial positions
    initialPositions = additionalStartingPositionsList
    #randomInitialPositionsNumber = int(proportionOfRandomInitialPositions * numberOfInitialPositions)
    selfPlayInitialPositions = int( (1 - proportionOfRandomInitialPositions) * numberOfInitialPositions)
    
    # Initial positions obtained through self-play
    #while len(initialPositions) < numberOfInitialPositions:
    createdSelfPlayInitialPositions = 0
    while createdSelfPlayInitialPositions < selfPlayInitialPositions:
        numberOfMoves = random.randint(0, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1 # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            chosenMoveTensor = neuralNetwork.ChooseAMove(positionTensor, playerList[0], authority,
                                                         preApplySoftMax=True, softMaxTemperature=1.0,
                                                         epsilon=epsilon)
            positionTensor, winner = authority.Move(positionTensor, playerList[0], chosenMoveTensor)
            moveNdx += 1
            positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
        if winner is None:
            initialPositions.append(positionTensor.clone())
            createdSelfPlayInitialPositions += 1

    while len(initialPositions) < numberOfInitialPositions: # Complete with random games
        numberOfMoves = random.randint(0, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1  # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            player = playerList[moveNdx % 2]
            # print ("GenerateMoveStatistics(): player = {}".format(player))
            randomMoveTensor = ChooseARandomMove(positionTensor, player, authority)
            positionTensor, winner = authority.Move(positionTensor, player, randomMoveTensor)
            moveNdx += 1
        if winner is None:
            initialPositions.append(positionTensor.clone())


    # For each initial position, evaluate the value of each possible move through self-play
    positionMoveStatistics = list()
    for initialPosition in initialPositions:
        (averageValuesTensor, standardDeviationTensor, legalMovesNMask) = \
        PositionExpectedMoveValues(
            playerList,
            authority,
            neuralNetwork,
            initialPosition,
            numberOfGamesForEvaluation,
            softMaxTemperatureForSelfPlayEvaluation,
            epsilon
        )
        positionMoveStatistics.append((initialPosition, averageValuesTensor,
                                      standardDeviationTensor, legalMovesNMask))

    return positionMoveStatistics
"""
def GenerateMoveStatistics(playerList,
                            authority,
                            neuralNetwork,
                            proportionOfRandomInitialPositions,
                            maximumNumberOfMovesForInitialPositions,
                            numberOfInitialPositions,
                            numberOfGamesForEvaluation,
                            softMaxTemperatureForSelfPlayEvaluation,
                            epsilon,
                            depthOfExhaustiveSearch,
                            additionalStartingPositionsList=[]
                            ):
    # Create initial positions
    initialPositions = additionalStartingPositionsList
    selfPlayInitialPositions = int( (1 - proportionOfRandomInitialPositions) * numberOfInitialPositions)

    # Initial positions obtained through self-play
    createdSelfPlayInitialPositions = 0
    while createdSelfPlayInitialPositions < selfPlayInitialPositions:
        numberOfMoves = random.randint(0, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1 # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            chosenMoveTensor = neuralNetwork.ChooseAMove(positionTensor, playerList[0], authority,
                                                         preApplySoftMax=True, softMaxTemperature=1.0,
                                                         epsilon=epsilon)
            positionTensor, winner = authority.Move(positionTensor, playerList[0], chosenMoveTensor)
            moveNdx += 1
            positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
        if winner is None:
            initialPositions.append(positionTensor.clone())
            createdSelfPlayInitialPositions += 1

    while len(initialPositions) < numberOfInitialPositions: # Complete with random games
        numberOfMoves = random.randint(0, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1  # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            player = playerList[moveNdx % 2]
            # print ("GenerateMoveStatistics(): player = {}".format(player))
            randomMoveTensor = ChooseARandomMove(positionTensor, player, authority)
            positionTensor, winner = authority.Move(positionTensor, player, randomMoveTensor)
            moveNdx += 1
        if winner is None:
            initialPositions.append(positionTensor.clone())


    # For each initial position, evaluate the value of each possible move through self-play
    positionMoveStatistics = list()
    for initialPosition in initialPositions:
        (averageValuesTensor, standardDeviationTensor, legalMovesNMask) = \
        PositionExpectedMoveValues(
            playerList,
            authority,
            neuralNetwork,
            initialPosition,
            numberOfGamesForEvaluation,
            softMaxTemperatureForSelfPlayEvaluation,
            epsilon,
            depthOfExhaustiveSearch
        )
        positionMoveStatistics.append((initialPosition, averageValuesTensor,
                                      standardDeviationTensor, legalMovesNMask))

    return positionMoveStatistics

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