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
import expectedMoveValues

def GenerateMoveStatistics(playerList,
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
                            resultsQueue = None, # For multiprocessing
                            event=None, # For multiprocessing
                            processNdx=None # For multiprocessing
                            ):
    print("generateMoveStatistics.GenerateMoveStatistics(): len(additionalStartingPositionsList) = {}".format(len(additionalStartingPositionsList)))
    # Create initial positions
    initialPositions = additionalStartingPositionsList
    selfPlayInitialPositions = numberOfInitialPositions#int( (1 - proportionOfRandomInitialPositions) * numberOfInitialPositions)

    minimumNumberOfMovesForInitialPositions = numberOfMovesForInitialPositionsMinMax[0]
    maximumNumberOfMovesForInitialPositions = numberOfMovesForInitialPositionsMinMax[1]

    if minimumNumberOfMovesForInitialPositions > maximumNumberOfMovesForInitialPositions:
        temp = minimumNumberOfMovesForInitialPositions
        minimumNumberOfMovesForInitialPositions = maximumNumberOfMovesForInitialPositions
        maximumNumberOfMovesForInitialPositions = temp

    # Initial positions obtained through self-play
    createdSelfPlayInitialPositions = 0
    while createdSelfPlayInitialPositions < selfPlayInitialPositions:
        numberOfMoves = random.randint(minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1 # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            chosenMoveTensor = neuralNetwork.ChooseAMove(positionTensor, playerList[0], authority,
                                                         chooseHighestProbabilityIfAtLeast=chooseHighestProbabilityIfAtLeast,
                                                         preApplySoftMax=True, softMaxTemperature=1.0,
                                                         epsilon=epsilon)
            positionTensor, winner = authority.Move(positionTensor, playerList[0], chosenMoveTensor)
            moveNdx += 1
            positionTensor = authority.SwapPositions(positionTensor, playerList[0], playerList[1])
        if winner is None:
            initialPositions.append(positionTensor.clone())
            createdSelfPlayInitialPositions += 1

    while len(initialPositions) < numberOfInitialPositions: # Complete with random games
        numberOfMoves = random.randint(minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1  # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            player = playerList[moveNdx % 2]
            # print ("GenerateMoveStatistics(): player = {}".format(player))
            randomMoveTensor = utilitiies.ChooseARandomMove(positionTensor, player, authority)
            positionTensor, winner = authority.Move(positionTensor, player, randomMoveTensor)
            moveNdx += 1
        if winner is None:
            initialPositions.append(positionTensor.clone())


    # For each initial position, evaluate the value of each possible move through self-play
    positionMoveStatistics = list()
    for initialPosition in initialPositions:
        (averageValuesTensor, standardDeviationTensor, legalMovesNMask) = \
        expectedMoveValues.PositionExpectedMoveValues(
            playerList,
            authority,
            neuralNetwork,
            chooseHighestProbabilityIfAtLeast,
            initialPosition,
            numberOfGamesForEvaluation,
            softMaxTemperatureForSelfPlayEvaluation,
            epsilon,
            depthOfExhaustiveSearch
        )
        positionMoveStatistics.append((initialPosition, averageValuesTensor,
                                      standardDeviationTensor, legalMovesNMask))

    if resultsQueue is not None and event is not None and processNdx is not None:
        print ("generateMoveStatistics.GenerateMoveStatistics(): len(positionMoveStatistics) = {}".format(len(positionMoveStatistics) ))
        print ("generateMoveStatistics.GenerateMoveStatistics(): putting positionMoveStatistics:")
        resultsQueue.put((processNdx, positionMoveStatistics))
        event.wait()
    else:
        return positionMoveStatistics

def GenerateMoveStatisticsMultiprocessing(
                            playerList,
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
                            ):
    print ("generateMoveStatistics.GenerateMoveStatisticsMultiprocessing()")
    print ("generateMoveStatistics.GenerateMoveStatisticsMultiprocessing() len(additionalStartingPositionsList) = {}".format(len(additionalStartingPositionsList)) )
    # Distribute the additional starting positions in separate lists
    additionalStartingPositionsListList = [[] for processNdx in range(numberOfProcesses)]
    for additionalStartingPositionNdx in range(len(additionalStartingPositionsList)):
        chosenProcessNdx = additionalStartingPositionNdx % numberOfProcesses
        additionalStartingPositionsListList[chosenProcessNdx].append(
            additionalStartingPositionsList[additionalStartingPositionNdx])

    print ("generateMoveStatistics.GenerateMoveStatisticsMultiprocessing(): additionalStartingPositionsListList = {}".format(additionalStartingPositionsListList))
    # Number of initial positions per process
    numberOfInitialPositionsList = []
    for processNdx in range(numberOfProcesses):
        numberOfInitialPositionsList.append(len(additionalStartingPositionsListList[processNdx]) )
    runningProcessNdx = 0
    while sum(numberOfInitialPositionsList) < numberOfInitialPositions:
        runningProcessNdx = runningProcessNdx % numberOfProcesses
        numberOfInitialPositionsList[runningProcessNdx] += 1
        runningProcessNdx += 1
    print ("generateMoveStatistics.GenerateMoveStatisticsMultiprocessing(): numberOfInitialPositionsList = {}".format(numberOfInitialPositionsList))

    outputsList = []

    # Start the processes
    processList = []
    processNdxToEventDic = {}
    resultsQueue = torch.multiprocessing.Queue()
    for processNdx in range(numberOfProcesses):
        event = torch.multiprocessing.Event()
        process = torch.multiprocessing.Process(
            target=GenerateMoveStatistics,
            args=(
                playerList,
                authority,
                neuralNetwork,
                proportionOfRandomInitialPositions,
                numberOfMovesForInitialPositionsMinMax,
                numberOfInitialPositionsList[processNdx],
                numberOfGamesForEvaluation,
                softMaxTemperatureForSelfPlayEvaluation,
                epsilon,
                depthOfExhaustiveSearch,
                chooseHighestProbabilityIfAtLeast,
                additionalStartingPositionsListList[processNdx],
                resultsQueue,
                event,
                processNdx
            )
        )
        process.start()
        processList.append(process)
        processNdxToEventDic[processNdx] = event


    for resultNdx in range(numberOfProcesses):
        (processNdx, positionMovesStatistics) = resultsQueue.get()
        processNdxToEventDic[processNdx].set()
        outputsList += positionMovesStatistics

    for p in processList:
        p.join()
    return outputsList

def GenerateMoveStatisticsWithMiniMax(
                            playerList,
                            authority,
                            neuralNetwork,
                            numberOfMovesForInitialPositionsMinMax,
                            numberOfInitialPositions,
                            maximumDepthOfExhaustiveSearch,
                            additionalStartingPositionsList=[]
                            ):
    # Create initial positions
    initialPositions = additionalStartingPositionsList

    minimumNumberOfMovesForInitialPositions = numberOfMovesForInitialPositionsMinMax[0]
    maximumNumberOfMovesForInitialPositions = numberOfMovesForInitialPositionsMinMax[1]

    if minimumNumberOfMovesForInitialPositions > maximumNumberOfMovesForInitialPositions:
        temp = minimumNumberOfMovesForInitialPositions
        minimumNumberOfMovesForInitialPositions = maximumNumberOfMovesForInitialPositions
        maximumNumberOfMovesForInitialPositions = temp

    while len(initialPositions) < numberOfInitialPositions: # Complete with random games
        numberOfMoves = random.randint(minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions)
        if numberOfMoves % 2 == 1:
            numberOfMoves += 1  # Make sure the last player to have played is playerList[1]
        positionTensor = authority.InitialPosition()
        winner = None
        moveNdx = 0
        while moveNdx < numberOfMoves and winner is None:
            player = playerList[moveNdx % 2]
            # print ("GenerateMoveStatistics(): player = {}".format(player))
            randomMoveTensor = utilities.ChooseARandomMove(positionTensor, player, authority)
            positionTensor, winner = authority.Move(positionTensor, player, randomMoveTensor)
            moveNdx += 1
        if winner is None:
            initialPositions.append(positionTensor.clone())


    # For each initial position, evaluate the value of each possible move through semi-exhaustive minimax
    positionMoveStatistics = list()
    for initialPosition in initialPositions:
        (averageValuesTensor, standardDeviationTensor, legalMovesNMask) = \
        expectedMoveValues.SemiExhaustiveMiniMax(
            playerList,
            authority,
            neuralNetwork,
            #0, # Always choose the highest value move
            initialPosition,
            epsilon=0,
            maximumDepthOfSemiExhaustiveSearch=maximumDepthOfExhaustiveSearch,
            currentDepth=1,
            numberOfTopMovesToDevelop=-1 # Develop all moves => exhaustive search
        )
        positionMoveStatistics.append((initialPosition, averageValuesTensor,
                                      standardDeviationTensor, legalMovesNMask))

    return positionMoveStatistics