import torch
import policy
import reach100
import random
import time
import multiprocessing


playerList = ['Player1', 'Player2']


def main():
    print ("learnReach100.py main()")
    proportionOfRandomInitialPositions = 0.5
    maximumNumberOfMovesForInitialPositions = 20
    numberOfInitialPositions = 10
    numberOfGamesForEvaluation = 10

    authority = reach100.Authority()
    #positionTensor = authority.InitialPosition()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()

    neuralNetwork = policy.NeuralNetwork(positionTensorShape,
                                         '[(7, 1, 1, 32), (7, 1, 1, 30)]',
                                         moveTensorShape)

    startTime = time.time()
    positionToMoveProbabilitiesAndValueDic = policy.GeneratePositionToMoveProbabilityAndValueDic(
        playerList, authority, neuralNetwork,
        proportionOfRandomInitialPositions,
        maximumNumberOfMovesForInitialPositions,
        numberOfInitialPositions,
        numberOfGamesForEvaluation,
    )
    endTime = time.time()
    print ("main(): Delay for GeneratePositionToMoveProbabilityAndValueDic(): {} s".format(endTime - startTime))
    for position, probabilitiesValue in positionToMoveProbabilitiesAndValueDic.items():
        currentSum = authority.CurrentSum(position)
        print ("currentSum = {}".format(currentSum))
        print ("Move probabilities: \n{}".format(probabilitiesValue[0]))
        print ("Value: {}".format(probabilitiesValue[1]))


if __name__ == '__main__':
    main()