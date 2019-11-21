import abc
import numpy
import utilities

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
    if nextPlayer == None:
        playersList[0]

    moveTensorShape = gameAuthority.MoveTensorShape()
    positionsList = [startingPosition]
    currentPosition = startingPosition
    winner = None
    while winner is None:
        if nextPlayer == playersList[1]:
            currentPosition = gameAuthority.SwapPosition(currentPosition, playersList[0], playersList[1])
        randomNbr = numpy.random.rand()
        if randomNbr < epsilon:
            chosenMoveTensor = utilities.ChooseARandomMove(currentPosition, playersList[0], gameAuthority)
        else:
            legalMovesMask = gameAuthority.LegalMovesMask(currentPosition, playersList[0])
            nonZeroCoordsTensor = legalMovesMask.nonzero()
            #print ("nonZeroCoordsTensor = {}".format(nonZeroCoordsTensor))
            chosenMoveTensor = None
            highestValue = -2.0
            for candidateMoveNdx in nonZeroCoordsTensor.shape[0]:
                candidateMoveTensor = torch.zeros(moveTensorShape)
                nonZeroCoords = nonZeroCoordsTensor[candidateMoveNdx]
                candidateMoveTensor[nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3] ] = 1.0
                candidatePositionAfterMove, candidateWinner = gameAuthority.Move(currentPosition, playersList[0], candidateMoveTensor)
                candidatePositionValue = evaluator.Value(candidatePositionAfterMove)
                if candidateWinner == playersList[0]:
                    candidatePositionValue = 1.0
                elif candidateWinner == positionsList[1]:
                    candidatePositionValue = -1.0
                elif candidateWinner == 'draw':
                    candidatePositionValue = 0

                if candidatePositionValue > highestValue:
                    highestValue = candidatePositionValue
                    chosenMoveTensor = candidateMoveTensor

        currentPosition, winner = gameAuthority.Move(currentPosition, playersList[0], chosenMoveTensor)
        if nextPlayer == playersList[1]: # De-swap, reverse the winner
            currentPosition = gameAuthority.SwapPosition(currentPosition, playersList[0], playersList[1])
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