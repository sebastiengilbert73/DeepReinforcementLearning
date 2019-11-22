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
                            processNdx=None, # For multiprocessing
                            maximumNumberOfMoves=100
                            ):
    #print("generateMoveStatistics.GenerateMoveStatistics(): len(additionalStartingPositionsList) = {}".format(len(additionalStartingPositionsList)))
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
            depthOfExhaustiveSearch,
            maximumNumberOfMoves
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

def GenerateEndGameStatistics(
        playerList,
        authority,
        neuralNetwork,
        keepNumberOfMovesBeforeEndGame,
        numberOfPositions,
        numberOfGamesForEvaluation,
        softMaxTemperatureForSelfPlayEvaluation,
        epsilon,
        maximumNumberOfMovesForFullGameSimulation,
        maximumNumberOfMovesForEndGameSimulation,
        additionalStartingPositionsList=[],
        resultsQueue = None, # For multiprocessing
        event=None, # For multiprocessing
        processNdx=None # For multiprocessing
        ):
    positionMovesStatistics = []

    #for positionNdx in range(numberOfPositions):
    while len(positionMovesStatistics) < numberOfPositions:
        positionsList, winner = expectedMoveValues.SimulateAGame(
            playerList,
            authority,
            neuralNetwork,
            softMaxTemperatureForSelfPlayEvaluation,
            epsilon,
            maximumNumberOfMovesForFullGameSimulation
        )
        if len(positionsList) < maximumNumberOfMovesForFullGameSimulation:
            selectedNdx = len(positionsList) - 1 - keepNumberOfMovesBeforeEndGame
            if selectedNdx %2 == 1: # Select a position where it's player0's turn
                selectedNdx -= 1
            selectedNdx = max(selectedNdx, 0)
            #print ("GenerateEndGameStatistics(): selectedNdx = {}".format(selectedNdx))
            selectedPosition = positionsList[selectedNdx]
            #print ("GenerateEndGameStatistics(): selectedPosition = \n")
            #authority.Display(selectedPosition)

            positionMoveStatistics = GenerateMoveStatistics(playerList,
                                   authority,
                                   neuralNetwork,
                                   proportionOfRandomInitialPositions=0,
                                   numberOfMovesForInitialPositionsMinMax=(0, 0),
                                   numberOfInitialPositions=0,
                                   numberOfGamesForEvaluation=numberOfGamesForEvaluation,
                                   softMaxTemperatureForSelfPlayEvaluation=softMaxTemperatureForSelfPlayEvaluation,
                                   epsilon=epsilon,
                                   depthOfExhaustiveSearch=1,
                                   chooseHighestProbabilityIfAtLeast=1,
                                   additionalStartingPositionsList=[selectedPosition],
                                   #resultsQueue=None,  # For multiprocessing
                                   #event=None,  # For multiprocessing
                                   #processNdx=None  # For multiprocessing
                                   maximumNumberOfMoves=maximumNumberOfMovesForEndGameSimulation
                                   )
            positionMovesStatistics.append(positionMoveStatistics[0])
            #print ("GenerateEndGameStatistics(): len(positionMoveStatistics[0]) = {}; len(positionMovesStatistics) = {}".format(len(positionMoveStatistics[0]), len(positionMovesStatistics)))
            #print ("GenerateEndGameStatistics(): positionMoveStatistics = \n{}".format(positionMoveStatistics))





    #print ("GenerateEndGameStatistics(): positionsList = \n{}".format(positionsList))
    #print ("GenerateEndGameStatistics(): len(positionsList) = \n{}".format(len(positionsList)))

    return positionMovesStatistics




def main():
    import checkers
    import moveEvaluation.ConvolutionStack
    print ("generateMoveStatistics.py main()")

    authority = checkers.Authority()
    playerList = authority.PlayersList()
    #neuralNetwork = moveEvaluation.ConvolutionStack.Net()
    #neuralNetwork.Load("/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(6,1,8,8)_[(5,32),(5,32),(5,32)]_(4,1,8,8)_checkers_48.pth")
    keepNumberOfMovesBeforeEndGame = 3
    numberOfPositions = 3
    numberOfGamesForEvaluation = 5
    softMaxTemperatureForSelfPlayEvaluation = 0.1
    epsilon = 0.1
    maximumNumberOfMovesForFullGameSimulation = 100
    maximumNumberOfMovesForEndGameSimulation = 10

    positionMovesStatistics = GenerateEndGameStatistics(
        playerList,
        authority,
        None, #neuralNetwork,
        keepNumberOfMovesBeforeEndGame,
        numberOfPositions,
        numberOfGamesForEvaluation,
        softMaxTemperatureForSelfPlayEvaluation,
        epsilon,
        maximumNumberOfMovesForFullGameSimulation,
        maximumNumberOfMovesForEndGameSimulation
    )

    print ("positionMovesStatistics = \n{}".format(positionMovesStatistics))

if __name__ == '__main__':
    main()