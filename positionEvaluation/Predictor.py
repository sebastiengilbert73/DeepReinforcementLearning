import abc
import numpy
import utilities
import torch
import pickle

class Evaluator(abc.ABC):
    """
    Abstract class that predicts the value of a position, when it is the opponent's turn
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def Value(self, position):
        pass # return a float

    @abc.abstractmethod
    def LearnFromMinibatch(self, minibatchFeaturesTensor, minibatchTargetValues):
        pass

    def Save(self, filepath):
        binary_file = open(filepath, mode='wb')
        pickle.dump(self, binary_file)
        binary_file.close()


def Load(filepath):
    pickle_in = open(filepath, 'rb')
    evaluator = pickle.load(pickle_in)
    pickle_in.close()
    return evaluator

def SimulateAGame(evaluator, gameAuthority, startingPosition=None, nextPlayer=None, epsilon=0.1):
    playersList = gameAuthority.PlayersList()
    if startingPosition is None:
        startingPosition = gameAuthority.InitialPosition()
    if nextPlayer is None:
        nextPlayer = playersList[0]

    moveTensorShape = gameAuthority.MoveTensorShape()
    positionsList = [startingPosition]
    currentPosition = startingPosition
    winner = None
    while winner is None:
        #print ("SimulateAGame(): nextPlayer = {}".format(nextPlayer))
        if nextPlayer == playersList[1]:
            currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
            #print("SimulateAGame(): playersList[1] turn!")
        randomNbr = numpy.random.rand()
        if randomNbr < epsilon:
            chosenMoveTensor = utilities.ChooseARandomMove(currentPosition, playersList[0], gameAuthority)
        else:
            legalMovesMask = gameAuthority.LegalMovesMask(currentPosition, playersList[0])
            nonZeroCoordsTensor = legalMovesMask.nonzero()
            #print ("nonZeroCoordsTensor = {}".format(nonZeroCoordsTensor))
            chosenMoveTensor = None
            highestValue = -1.0E9
            for candidateMoveNdx in range(nonZeroCoordsTensor.shape[0]):
                candidateMoveTensor = torch.zeros(moveTensorShape)
                nonZeroCoords = nonZeroCoordsTensor[candidateMoveNdx]
                candidateMoveTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ] = 1.0
                candidatePositionAfterMove, candidateWinner = gameAuthority.Move(currentPosition, playersList[0], candidateMoveTensor)
                candidatePositionValue = evaluator.Value(candidatePositionAfterMove.unsqueeze(0))[0]
                #print ("Predictor.py SimulateAGame(): candidatePositionValue = {}".format(candidatePositionValue))
                if candidateWinner == playersList[0]:
                    candidatePositionValue = 1.0
                elif candidateWinner == playersList[1]:
                    candidatePositionValue = -1.0
                elif candidateWinner == 'draw':
                    candidatePositionValue = 0

                if candidatePositionValue > highestValue:
                    highestValue = candidatePositionValue
                    chosenMoveTensor = candidateMoveTensor

        currentPosition, winner = gameAuthority.Move(currentPosition, playersList[0], chosenMoveTensor)
        if nextPlayer == playersList[1]: # De-swap, reverse the winner
            currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
            #print ("SimulateAGame(): playersList[1]: Swapped position after move: \n{}".format(currentPosition))
            if winner == playersList[0]:
                winner = playersList[1]
            elif winner == playersList[1]:
                winner = playersList[0]

        positionsList.append(currentPosition)

        if nextPlayer == playersList[0]:
            nextPlayer = playersList[1]
        else:
            nextPlayer = playersList[0]
    return positionsList, winner

def SimulateMultipleGamesAndGetStatistics(evaluator, gameAuthority, numberOfGames, startingPosition=None,
                                          nextPlayer=None, epsilon=0.1):
    numberOfWinsForPlayer0 = 0
    numberOfWinsForPlayer1 = 0
    numberOfDraws = 0
    playersList = gameAuthority.PlayersList()

    for gameNdx in range(numberOfGames):
        positionsList, winner = SimulateAGame(evaluator, gameAuthority, startingPosition, nextPlayer, epsilon)
        if winner == playersList[0]:
            numberOfWinsForPlayer0 += 1
        elif winner == playersList[1]:
            numberOfWinsForPlayer1 += 1
        elif winner == 'draw':
            numberOfDraws += 1
        else:
            raise ValueError("SimulateMultipleGamesAndGetStatistics(): Unknown winner '{}'".format(winner))
    return (numberOfWinsForPlayer0, numberOfWinsForPlayer1, numberOfDraws)

def ExpectedReward(evaluator, gameAuthority, numberOfGames, startingPosition=None,
                                          nextPlayer=None, epsilon=0.1):
    (numberOfWinsForPlayer0, numberOfWinsForPlayer1, numberOfDraws) = SimulateMultipleGamesAndGetStatistics(
        evaluator, gameAuthority, numberOfGames, startingPosition, nextPlayer, epsilon)
    return (1.0 * numberOfWinsForPlayer0 -1.0 * numberOfWinsForPlayer1)/numberOfGames

def SimulateGamesAgainstARandomPlayer(evaluator, gameAuthority, numberOfGames, gameToRewardDict=None):
    playersList = gameAuthority.PlayersList()
    moveTensorShape = gameAuthority.MoveTensorShape()
    numberOfWinsForEvaluator = 0
    numberOfWinsForRandomPlayer = 0
    numberOfDraws = 0
    if gameToRewardDict is None:
        gameToRewardDict = {}

    for gameNdx in range(numberOfGames):
        evaluatorPlayer = playersList[gameNdx % 2]

        winner = None
        currentPosition = gameAuthority.InitialPosition()
        moveNdx = 0
        positionsList = []
        positionsList.append(currentPosition)
        while winner is None:
            nextPlayer = playersList[moveNdx % 2]
            chosenMoveTensor = None
            if nextPlayer == evaluatorPlayer:
                if nextPlayer == playersList[1]:
                    currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])

                legalMovesMask = gameAuthority.LegalMovesMask(currentPosition, playersList[0])
                nonZeroCoordsTensor = legalMovesMask.nonzero()
                highestValue = -1.0E9
                for candidateMoveNdx in range(nonZeroCoordsTensor.shape[0]):
                    candidateMoveTensor = torch.zeros(moveTensorShape)
                    nonZeroCoords = nonZeroCoordsTensor[candidateMoveNdx]
                    candidateMoveTensor[
                        nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
                    candidatePositionAfterMove, candidateWinner = gameAuthority.Move(currentPosition,
                                                                                     playersList[0],
                                                                                     candidateMoveTensor)
                    candidatePositionValue = evaluator.Value(candidatePositionAfterMove.unsqueeze(0))[0]
                    if candidateWinner == playersList[0]:
                        candidatePositionValue = 1.0
                    elif candidateWinner == playersList[1]:
                        candidatePositionValue = -1.0
                    elif candidateWinner == 'draw':
                        candidatePositionValue = 0

                    if candidatePositionValue > highestValue:
                        highestValue = candidatePositionValue
                        chosenMoveTensor = candidateMoveTensor

                currentPosition, winner = gameAuthority.Move(currentPosition, playersList[0], chosenMoveTensor)
                if nextPlayer == playersList[1]:  # De-swap, reverse the winner
                    currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
                    if winner == playersList[0]:
                        winner = playersList[1]
                    elif winner == playersList[1]:
                        winner = playersList[0]
                positionsList.append(currentPosition)

            else: # Random player's turn
                chosenMoveTensor = utilities.ChooseARandomMove(currentPosition, nextPlayer, gameAuthority)
                currentPosition, winner = gameAuthority.Move(currentPosition, nextPlayer, chosenMoveTensor)
                positionsList.append(currentPosition)

            moveNdx += 1
        if winner == evaluatorPlayer:
            numberOfWinsForEvaluator += 1
            gameToRewardDict[tuple(positionsList)] = 1.0
        elif winner == 'draw':
            numberOfDraws += 1
            gameToRewardDict[tuple(positionsList)] = 0.0
        else:
            numberOfWinsForRandomPlayer += 1
            gameToRewardDict[tuple(positionsList)] = -1.0

    else:
        return (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer, numberOfDraws)

def LegalMoveToExpectedReward(evaluator, gameAuthority, currentPosition, nextPlayer, numberOfGames, epsilon):
    legalMovesMask = gameAuthority.LegalMovesMask(currentPosition, nextPlayer)
    nonZeroCoordsTensor = legalMovesMask.nonzero()
    moveTensorShape = gameAuthority.MoveTensorShape()
    playersList = gameAuthority.PlayersList()
    legalMoveToExpectedRewardDict = {}

    for candidateMoveNdx in range(nonZeroCoordsTensor.shape[0]):
        candidateMoveTensor = torch.zeros(moveTensorShape)
        nonZeroCoords = nonZeroCoordsTensor[candidateMoveNdx]
        candidateMoveTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
        candidatePositionAfterMove, candidateWinner = gameAuthority.Move(currentPosition,
                                                                         nextPlayer,
                                                                         candidateMoveTensor)
        if nextPlayer == playersList[0]:
            nextPlayer = playersList[1]
        else:
            nextPlayer = playersList[0]
        expectedReward = ExpectedReward(evaluator, gameAuthority, numberOfGames, startingPosition=candidatePositionAfterMove,
                                          nextPlayer=nextPlayer, epsilon=epsilon)
        legalMoveToExpectedRewardDict[candidateMoveTensor] = expectedReward
    return legalMoveToExpectedRewardDict

def SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions,
                        numberOfPositions):
    selectedPositionsList = []
    while len(selectedPositionsList) < numberOfPositions:
        gamePositionsList, winner = SimulateAGame(
            evaluator=None, gameAuthority=authority, startingPosition=None, nextPlayer=None, epsilon=1.0) # With epsilon=1.0, the evaluator will never be called
        if len(gamePositionsList) >= minimumNumberOfMovesForInitialPositions:
            maxNdx = min(maximumNumberOfMovesForInitialPositions - 1, len(gamePositionsList) - 2) # The last index cannot be the last position, since the game is over
            if maxNdx < 0:
                maxNdx = 0
            selectedNdx = numpy.random.randint(maxNdx + 1) # maxNdx can be selected: {0, 1, 2, ..., maxNdx}
            selectedPositionsList.append(gamePositionsList[selectedNdx])
    return selectedPositionsList