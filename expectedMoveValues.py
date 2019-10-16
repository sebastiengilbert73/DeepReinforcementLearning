import argparse
import ast
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# To avoid `RuntimeError: received 0 items of ancdata` Cf. https://github.com/pytorch/pytorch/issues/973
import math
import random
import sys
import statistics
import numpy
import utilities

def SimulateGameAndGetReward(playerList,
                             positionTensor,
                             nextPlayer,
                             authority,
                             neuralNetwork, # If None, do random moves
                             chooseHighestProbabilityIfAtLeast,
                             preApplySoftMax,
                             softMaxTemperature,
                             epsilon):
    winner = None
    if nextPlayer == playerList[0]:
        moveNdx = 0
    elif nextPlayer == playerList[1]:
        moveNdx = 1
    else:
        raise ValueError("expectedMoveValues.SimulateGameAndGetReward(): Unknown player '{}'".format(nextPlayer))
    if nextPlayer == playerList[1]:
        positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
    while winner is None:
        #print ("SimulateGameAndGetReward(): positionTensor = {}".format(positionTensor))
        player = playerList[moveNdx % 2]
        if neuralNetwork is None:
            chosenMoveTensor = utilities.ChooseARandomMove(positionTensor, playerList[0], authority)
        else:
            #print ("SimulateGameAndGetReward(): player = {}; positionTensor = \n{}".format(player, positionTensor))
            chosenMoveTensor = neuralNetwork.ChooseAMove(
                positionTensor,
                playerList[0],
                authority,
                chooseHighestProbabilityIfAtLeast,
                preApplySoftMax,
                softMaxTemperature,
                epsilon=epsilon
            )
        if chosenMoveTensor is not None:
            chosenMoveTensor = chosenMoveTensor.detach()
        #print ("SimulateGameAndGetReward(): chosenMoveTensor =\n{}".format(chosenMoveTensor))
        if chosenMoveTensor is not None:
            positionTensor, winner = authority.Move(positionTensor, playerList[0], chosenMoveTensor)
        #winner = authority.MoveInPlace(positionTensor, playerList[0], chosenMoveTensor)
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

def PositionExpectedMoveValues(
        playerList,
        authority,
        neuralNetwork,
        chooseHighestProbabilityIfAtLeast,
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
                chooseHighestProbabilityIfAtLeast,
                neuralNetwork,
                softMaxTemperatureForSelfPlayEvaluation,
                epsilon,
                numberOfGamesForEvaluation
            )
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardAverage
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardStandardDeviation
        #print ("PositionExpectedMoveValues2(): high-level moveValuesTensor = {}".format(moveValuesTensor))

    return moveValuesTensor, standardDeviationTensor, legalMovesMask

def SemiExhaustiveExpectedMoveValues(
        playerList,
        authority,
        neuralNetwork,
        chooseHighestProbabilityIfAtLeast,
        position,
        numberOfGamesForEvaluation,
        softMaxTemperatureForSelfPlayEvaluation,
        epsilon,
        maximumDepthOfSemiExhaustiveSearch,
        currentDepth,
        numberOfTopMovesToDevelop
        ):

    legalMovesMask = authority.LegalMovesMask(position, playerList[0])
    moveTensorShape = authority.MoveTensorShape()

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    moveValuesTensor = torch.zeros(moveTensorShape)
    standardDeviationTensor = torch.zeros(moveTensorShape) - 1.0

    neuralNetworkOutput = neuralNetwork(position.unsqueeze(0)).squeeze(0)
    #print ("SemiExhaustiveExpectedMoveValues(): neuralNetworkOutput = \n{}".format(neuralNetworkOutput))

    nonZeroCoordsToNetOutputDic = dict()
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        #print ("SemiExhaustiveExpectedMoveValues(): nonZeroCoords = {}".format(nonZeroCoords))
        nonZeroCoordsToNetOutputDic[nonZeroCoords] = neuralNetworkOutput[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ].item()

    #print ("SemiExhaustiveExpectedMoveValues(): nonZeroCoordsToNetOutputDic = \n{}".format(nonZeroCoordsToNetOutputDic))
    legalMoveValuesList = list(nonZeroCoordsToNetOutputDic.values())
    legalMoveValuesList.sort(reverse=True)
    #print ("SemiExhaustiveExpectedMoveValues(): legalMoveValuesList = {}".format(legalMoveValuesList))
    minimumValueForExhaustiveSearch = -2.0
    if numberOfTopMovesToDevelop > 0 and numberOfTopMovesToDevelop < len(legalMoveValuesList):
        minimumValueForExhaustiveSearch = legalMoveValuesList[numberOfTopMovesToDevelop - 1]
    elif numberOfTopMovesToDevelop < 1:
        minimumValueForExhaustiveSearch = 2.0
    if currentDepth > maximumDepthOfSemiExhaustiveSearch:
        minimumValueForExhaustiveSearch = 2.0
    #print ("SemiExhaustiveExpectedMoveValues(): minimumValueForExhaustiveSearch = {}".format(minimumValueForExhaustiveSearch))

    # Go through the legal moves
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        #print ("nonZeroCoordsNdx = {}".format(nonZeroCoordsNdx))
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        weirdMoveFlag = False#\
            #nonZeroCoords[0] == 0 and \
            #nonZeroCoords[1] == 0 and \
            #nonZeroCoords[2] == 2 and \
            #nonZeroCoords[3] == 0

        firstMoveArr = numpy.zeros(moveTensorShape)
        firstMoveArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        firstMoveTensor = torch.from_numpy(firstMoveArr).float()
        positionAfterFirstMoveTensor, winner = authority.Move(position, playerList[0],
                                                              firstMoveTensor)
        if winner is playerList[0]:
            #print ("{} won!".format(playerList[0]))
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            continue # Go back to the beginning of the for body
        elif winner is playerList[1]:
            #print ("{} won!".format(playerList[1]))
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = -1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            continue  # Go back to the beginning of the for body
        elif winner is 'draw':
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            continue  # Go back to the beginning of the for body

        #print ("After checking the winner")
        moveValue = neuralNetworkOutput[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ].item()
        #print ("SemiExhaustiveExpectedMoveValues(): nonZeroCoords = {}; moveValue = {}".format(nonZeroCoords, moveValue))
        if moveValue >= minimumValueForExhaustiveSearch:
            #print ("Exhaustive search")
            # Swap the positions to present the situation as playerList[0] to the neural network
            swappedPosition = authority.SwapPositions(positionAfterFirstMoveTensor,
                                                      playerList[0], playerList[1])
            nextValuesTensor, nextStandardDeviationTensor, nextLegalMovesMask = \
            SemiExhaustiveExpectedMoveValues(
                playerList,
                authority,
                neuralNetwork,
                chooseHighestProbabilityIfAtLeast,
                swappedPosition,
                numberOfGamesForEvaluation,
                softMaxTemperatureForSelfPlayEvaluation,
                epsilon,
                maximumDepthOfSemiExhaustiveSearch,
                currentDepth + 1,
                numberOfTopMovesToDevelop
            )
            # Find the highest value in the legal moves
            nextHighestReward = -2.0
            nextHighestRewardCoords = (0, 0, 0, 0)
            nextNonZeroCoordsTensor = torch.nonzero(nextLegalMovesMask)
            for nextNonZeroCoordsNdx in range(nextNonZeroCoordsTensor.size(0)):
                nextNonZeroCoords = nextNonZeroCoordsTensor[nextNonZeroCoordsNdx]
                if nextValuesTensor[
                    nextNonZeroCoords[0], nextNonZeroCoords[1], nextNonZeroCoords[2], nextNonZeroCoords[3] ] \
                        > nextHighestReward:
                    nextHighestReward = nextValuesTensor[
                        nextNonZeroCoords[0], nextNonZeroCoords[1], nextNonZeroCoords[2], nextNonZeroCoords[3] ]
                    nextHighestRewardCoords = nextNonZeroCoords

            #print ("SemiExhaustiveExpectedMoveValues(): nextValuesTensor = \n{}\nnextStandardDeviationTensor =\n{}\nnextLegalMovesMask =\n{}".format(nextValuesTensor, nextStandardDeviationTensor, nextLegalMovesMask))
            # The opponent will choose the highest average reward
            correspondingStdDeviation = nextStandardDeviationTensor[
                nextHighestRewardCoords[0], nextHighestRewardCoords[1], nextHighestRewardCoords[2], nextHighestRewardCoords[3]
            ]
            # The actual average reward for the current player is -1 x the average reward
            negatedReward = -1.0 * nextHighestReward
            #print ("negatedReward = {}; correspondingStdDeviation = {}".format(negatedReward, correspondingStdDeviation))
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = negatedReward
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = correspondingStdDeviation
        else:
            if weirdMoveFlag:
                print ("expectedMoveValues.SemiExhaustiveExpectedMoveValues(): weirdMoveFlag, Monte-Carlo Tree Search")
                print ("positionAfterFirstMoveTensor = \n{}".format(positionAfterFirstMoveTensor))
            (averageReward, rewardStandardDeviation) = \
            RewardStatistics(
                positionAfterFirstMoveTensor, currentDepth + 1, currentDepth,
                playerList, playerList[1],
                authority,
                chooseHighestProbabilityIfAtLeast,
                neuralNetwork,
                softMaxTemperatureForSelfPlayEvaluation,
                epsilon,
                numberOfGamesForEvaluation
            )
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = averageReward
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = rewardStandardDeviation
            if weirdMoveFlag:
                print ("averageReward = {}; rewardStandardDeviation = {}".format(averageReward, rewardStandardDeviation))
    return moveValuesTensor, standardDeviationTensor, legalMovesMask

def SemiExhaustiveMiniMax(
        playerList,
        authority,
        neuralNetwork,
        position,
        epsilon,
        maximumDepthOfSemiExhaustiveSearch,
        currentDepth,
        numberOfTopMovesToDevelop
    ):
    """
    :param playerList: The list of player names. Ex.: ['X', 'O']. It is assumed that the neural network is playing player0
    :param authority: The game authority. ex.: 'tictactoe'
    :param neuralNetwork: The neural network that will make decisions
    :param chooseHighestProbabilityIfAtLeast: If the highest probability is higher than this, automatically select it, instead of running a roulette
    :param position: The position tensor
    :param epsilon: The epsilon in the epsilon-greedy algorithm. Ex.: 0.1
    :param maximumDepthOfSemiExhaustiveSearch: The maximum depth of semi exhaustive search. Ex.: 2
    :param currentDepth: The current depth of semi exhaustive search. Ex.: 1
    :param numberOfTopMovesToDevelop: The number of top values for which we'll do exhaustive search. Ex.: 3
    :return: moveValuesTensor, standardDeviationTensor, legalMovesMask
    """

    legalMovesMask = authority.LegalMovesMask(position, playerList[0])
    moveTensorShape = authority.MoveTensorShape()

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    moveValuesTensor = torch.zeros(moveTensorShape)
    standardDeviationTensor = torch.zeros(moveTensorShape) - 1.0

    neuralNetworkOutput = neuralNetwork(position.unsqueeze(0)).squeeze(0)
    #print ("SemiExhaustiveSoftMax(): neuralNetworkOutput = \n{}".format(neuralNetworkOutput))

    nonZeroCoordsToNetOutputDic = dict()
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        #print ("SemiExhaustiveExpectedMoveValues(): nonZeroCoords = {}".format(nonZeroCoords))
        nonZeroCoordsToNetOutputDic[nonZeroCoords] = neuralNetworkOutput[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ].item()

    #print ("SemiExhaustiveSoftMax(): nonZeroCoordsToNetOutputDic = \n{}".format(nonZeroCoordsToNetOutputDic))

    legalMoveValuesList = list(nonZeroCoordsToNetOutputDic.values())
    legalMoveValuesList.sort(reverse=True)
    #print ("SemiExhaustiveExpectedMoveValues(): legalMoveValuesList = {}".format(legalMoveValuesList))
    minimumValueForExhaustiveSearch = -2.0
    if numberOfTopMovesToDevelop > 0 and numberOfTopMovesToDevelop < len(legalMoveValuesList):
        minimumValueForExhaustiveSearch = legalMoveValuesList[numberOfTopMovesToDevelop - 1]
    elif numberOfTopMovesToDevelop == 0: # Allow negative value to mean "all of them" => exhaustive search
        minimumValueForExhaustiveSearch = 2.0
    if currentDepth > maximumDepthOfSemiExhaustiveSearch:
        minimumValueForExhaustiveSearch = 2.0
    #print ("SemiExhaustiveSoftMax(): minimumValueForExhaustiveSearch = {}".format(minimumValueForExhaustiveSearch))

    # Go through the legal moves
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
        #print ("nonZeroCoordsNdx = {}".format(nonZeroCoordsNdx))
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        weirdMoveFlag = False#\
            #nonZeroCoords[0] == 0 and \
            #nonZeroCoords[1] == 0 and \
            #nonZeroCoords[2] == 2 and \
            #nonZeroCoords[3] == 0

        firstMoveArr = numpy.zeros(moveTensorShape)
        firstMoveArr[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1
        firstMoveTensor = torch.from_numpy(firstMoveArr).float()
        positionAfterFirstMoveTensor, winner = authority.Move(position, playerList[0],
                                                              firstMoveTensor)
        if winner is playerList[0]:
            #print ("{} won!".format(playerList[0]))
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            continue # Go back to the beginning of the for body
        elif winner is playerList[1]:
            #print ("{} won!".format(playerList[1]))
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = -1.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            continue  # Go back to the beginning of the for body
        elif winner is 'draw':
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0.0
            continue  # Go back to the beginning of the for body

        #print ("After checking the winner")
        moveValue = neuralNetworkOutput[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ].item()
        #print ("SemiExhaustiveExpectedMoveValues(): nonZeroCoords = {}; moveValue = {}".format(nonZeroCoords, moveValue))
        if moveValue >= minimumValueForExhaustiveSearch:
            #print ("Exhaustive search")
            # Swap the positions to present the situation as playerList[0] to the neural network
            swappedPosition = authority.SwapPositions(positionAfterFirstMoveTensor,
                                                      playerList[0], playerList[1])
            nextValuesTensor, nextStandardDeviationTensor, nextLegalMovesMask = \
            SemiExhaustiveMiniMax(
                playerList,
                authority,
                neuralNetwork,
                #chooseHighestProbabilityIfAtLeast,
                swappedPosition,
                epsilon,
                maximumDepthOfSemiExhaustiveSearch,
                currentDepth + 1,
                numberOfTopMovesToDevelop
            )
            # Find the highest value in the legal moves
            nextHighestReward = -2.0
            nextHighestRewardCoords = (0, 0, 0, 0)
            nextNonZeroCoordsTensor = torch.nonzero(nextLegalMovesMask)
            for nextNonZeroCoordsNdx in range(nextNonZeroCoordsTensor.size(0)):
                nextNonZeroCoords = nextNonZeroCoordsTensor[nextNonZeroCoordsNdx]
                if nextValuesTensor[
                    nextNonZeroCoords[0], nextNonZeroCoords[1], nextNonZeroCoords[2], nextNonZeroCoords[3] ] \
                        > nextHighestReward:
                    nextHighestReward = nextValuesTensor[
                        nextNonZeroCoords[0], nextNonZeroCoords[1], nextNonZeroCoords[2], nextNonZeroCoords[3] ]
                    nextHighestRewardCoords = nextNonZeroCoords

            #print ("SemiExhaustiveExpectedMoveValues(): nextValuesTensor = \n{}\nnextStandardDeviationTensor =\n{}\nnextLegalMovesMask =\n{}".format(nextValuesTensor, nextStandardDeviationTensor, nextLegalMovesMask))
            # The opponent will choose the highest average reward
            correspondingStdDeviation = nextStandardDeviationTensor[
                nextHighestRewardCoords[0], nextHighestRewardCoords[1], nextHighestRewardCoords[2], nextHighestRewardCoords[3]
            ]
            # The actual average reward for the current player is -1 x the average reward
            negatedReward = -1.0 * nextHighestReward
            #print ("negatedReward = {}; correspondingStdDeviation = {}".format(negatedReward, correspondingStdDeviation))
            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = negatedReward
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = correspondingStdDeviation
        else:
            if weirdMoveFlag:
                print ("expectedMoveValues.SemiExhaustiveMiniMax(): weirdMoveFlag, Monte-Carlo Tree Search")
                print ("positionAfterFirstMoveTensor = \n{}".format(positionAfterFirstMoveTensor))

            moveValuesTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = moveValue
            standardDeviationTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 0
            if weirdMoveFlag:
                print ("averageReward = {}; rewardStandardDeviation = {}".format(averageReward, rewardStandardDeviation))

    return moveValuesTensor, standardDeviationTensor, legalMovesMask

def RewardStatistics(positionTensor, searchDepth, maxSearchDepth, playersList, player, authority,
                  chooseHighestProbabilityIfAtLeast,
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
                chooseHighestProbabilityIfAtLeast,
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
                    chooseHighestProbabilityIfAtLeast,
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

def AverageRewardAgainstARandomPlayer(
        playerList,
        authority,
        neuralNetwork,  # If None, do random moves
        chooseHighestProbabilityIfAtLeast,
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
        # authority.Display(positionTensor)
        winner = None
        while winner is None:
            player = playerList[moveNdx % 2]
            if player == playerList[1] or neuralNetwork is None:
                chosenMoveTensor = utilities.ChooseARandomMove(positionTensor, player, authority)
            else:
                if moveChoiceMode == 'SoftMax':
                    chosenMoveTensor = neuralNetwork.ChooseAMove(
                        positionTensor,
                        player,
                        authority,
                        chooseHighestProbabilityIfAtLeast,
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
                            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[
                                3]] > highestValue:
                            highestValue = moveValuesTensor[
                                nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]]
                            highestValueCoords = nonZeroCoords
                    chosenMoveTensor[
                        highestValueCoords[0], highestValueCoords[1], highestValueCoords[2], highestValueCoords[
                            3]] = 1.0
                else:
                    raise NotImplementedError(
                        "AverageRewardAgainstARandomPlayer(): Unknown move choice mode '{}'".format(
                            moveChoiceMode))
            # print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
            positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
            # authority.Display(positionTensor)
            # print ("policy.AverageRewardAgainstARandomPlayer() winner = {}".format(winner))
            moveNdx += 1
            # positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
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
            numberOfLosses / numberOfGame)

def AverageRewardAgainstARandomPlayerKeepLosingGames(
                             playerList,
                             authority,
                             neuralNetwork, # If None, do random moves
                             chooseHighestProbabilityIfAtLeast,
                             preApplySoftMax,
                             softMaxTemperature,
                             numberOfGames,
                             moveChoiceMode='SoftMax',
                             numberOfGamesForMoveEvaluation=31,
                             depthOfExhaustiveSearch=2,
                             numberOfTopMovesToDevelop=3
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
                chosenMoveTensor = utilities.ChooseARandomMove(positionTensor, player, authority)
            else:
                if moveChoiceMode == 'SoftMax':
                    chosenMoveTensor = neuralNetwork.ChooseAMove(
                        positionTensor,
                        player,
                        authority,
                        chooseHighestProbabilityIfAtLeast,
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
                        chooseHighestProbabilityIfAtLeast,
                        positionTensor,
                        numberOfGamesForMoveEvaluation,
                        softMaxTemperature,
                        epsilon=0,
                        depthOfExhaustiveSearch=depthOfExhaustiveSearch
                    )
                    chosenMoveTensor = torch.zeros(authority.MoveTensorShape())
                    highestValue = -1E9
                    highestValueCoords = (0, 0, 0, 0)
                    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
                    if nonZeroCoordsTensor.size(0) == 0:
                        if authority.RaiseAnErrorIfNoLegalMove():
                            raise ValueError("AverageRewardAgainstARandomPlayerKeepLosingGames(): There is no legal move. positionTensor = \n{}".format(positionTensor))
                        chosenMoveTensor = None
                    else:
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
                elif moveChoiceMode == 'SemiExhaustiveExpectedMoveValues':
                    moveValuesTensor, standardDeviationTensor, legalMovesMask = \
                    SemiExhaustiveExpectedMoveValues(
                        playerList,
                        authority,
                        neuralNetwork,
                        chooseHighestProbabilityIfAtLeast,
                        positionTensor,
                        numberOfGamesForMoveEvaluation,
                        softMaxTemperature,
                        epsilon=0,
                        maximumDepthOfSemiExhaustiveSearch=depthOfExhaustiveSearch,
                        currentDepth=1,
                        numberOfTopMovesToDevelop=numberOfTopMovesToDevelop
                    )
                    chosenMoveTensor = torch.zeros(authority.MoveTensorShape())
                    highestValue = -1E9
                    highestValueCoords = (0, 0, 0, 0)
                    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
                    if nonZeroCoordsTensor.size(0) == 0:
                        if authority.RaiseAnErrorIfNoLegalMove():
                            raise ValueError("AverageRewardAgainstARandomPlayerKeepLosingGames(): There is no legal move. positionTensor = \n{}".format(positionTensor))
                        chosenMoveTensor = None
                    else:
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
                elif moveChoiceMode == 'SemiExhaustiveMiniMax':
                    moveValuesTensor, standardDeviationTensor, legalMovesMask = \
                    SemiExhaustiveMiniMax(
                        playerList,
                        authority,
                        neuralNetwork,
                        #chooseHighestProbabilityIfAtLeast,
                        positionTensor,
                        #softMaxTemperature,
                        epsilon=0,
                        maximumDepthOfSemiExhaustiveSearch=depthOfExhaustiveSearch,
                        currentDepth=1,
                        numberOfTopMovesToDevelop=numberOfTopMovesToDevelop
                    )
                    chosenMoveTensor = torch.zeros(authority.MoveTensorShape())
                    highestValue = -1E9
                    highestValueCoords = (0, 0, 0, 0)
                    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
                    if nonZeroCoordsTensor.size(0) == 0:
                        if authority.RaiseAnErrorIfNoLegalMove():
                            raise ValueError("AverageRewardAgainstARandomPlayerKeepLosingGames(): There is no legal move. positionTensor = \n{}".format(positionTensor))
                        chosenMoveTensor = None
                    else:
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
                    raise NotImplementedError("expectedMoveValues.AverageRewardAgainstARandomPlayerKeepLosingGames(): Unknown move choice mode '{}'".format(moveChoiceMode))
            # print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
            if chosenMoveTensor is not None:
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