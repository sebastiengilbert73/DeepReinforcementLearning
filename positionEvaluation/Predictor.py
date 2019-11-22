import abc
import numpy
import utilities
import torch

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