import argparse
import ast
import torch
import time
import random
import time
import policy
import moveEvaluation.ConvolutionStack
import tictactoe
import connect4

parser = argparse.ArgumentParser()
parser.add_argument('game', help='The game you want the neural networks to play')
parser.add_argument('neuralNetwork1', help="The filepath to the first neural network")
parser.add_argument('neuralNetwork2', help="The filepath to the second neural network")
parser.add_argument('--maximumDepthOfSemiExhaustiveSearch', help='The maximum depth of semi exhaustive search. Default: 2', type=int, default=2)
parser.add_argument('--numberOfTopMovesToDevelop', help='For semi-exhaustive search, the number of top moves to develop. Default: 3', type=int, default=3)
parser.add_argument('--softMaxTemperature', help='The softmax temperature, to add inject variance. Default: 0.3', type=float, default=0.3)
parser.add_argument('--numberOfGames', help='The number of games to play. Default: 300', type=int, default=300)
parser.add_argument('--displayPositions', action='store_true', help='Display positions')
args = parser.parse_args()


def AskTheNeuralNetworkToChooseAMove(
        playersList,
        authority,
        playingNeuralNetwork,
        positionTensor,
        depthOfExhaustiveSearch,
        numberOfTopMovesToDevelop,
        softMaxTemperature):

    moveValuesTensor, standardDeviationTensor, legalMovesMask = \
    policy.SemiExhaustiveMiniMax(
        playersList,
        authority,
        playingNeuralNetwork,
        positionTensor,
        0.0,
        depthOfExhaustiveSearch,
        1,
        numberOfTopMovesToDevelop
    )

    # Normalize probabilities
    normalizedActionValuesTensor = policy.NormalizeProbabilities(moveValuesTensor,
                                                                 legalMovesMask,
                                                                 preApplySoftMax=True,
                                                                 softMaxTemperature=softMaxTemperature)
    chosenMoveTensor = torch.zeros(authority.MoveTensorShape())
    # Choose with roulette
    runningSum = 0
    chosenCoordinates = None

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    randomNbr = random.random()
    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0) - 1):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        runningSum += normalizedActionValuesTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]
        ]
        if runningSum >= randomNbr and chosenCoordinates is None:
            chosenCoordinates = (nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3])
            break  # Stop looping
    if chosenCoordinates is None:  # and randomNbr - runningSum < 0.000001: # Choose the last candidate
        chosenNdx = nonZeroCoordsTensor.size(0) - 1
        nonZeroCoords = nonZeroCoordsTensor[chosenNdx]
        chosenCoordinates = (nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3])

    chosenMoveTensor[chosenCoordinates] = 1.0

    return chosenMoveTensor

def main():
    print ("neuralNetworksTournament.py main()")

    # Create the game authority
    if args.game == 'tictactoe':
        authority = tictactoe.Authority()
    elif args.game == 'connect4':
        authority = connect4.Authority()
    else:
        raise NotImplementedError("main(): unknown game '{}'".format(args.game))

    playersList = authority.PlayersList()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    neuralNetwork1 = moveEvaluation.ConvolutionStack.Net()
    neuralNetwork1.Load(args.neuralNetwork1)
    neuralNetwork2 = moveEvaluation.ConvolutionStack.Net()
    neuralNetwork2.Load(args.neuralNetwork2)
    neuralNetworks = [neuralNetwork1, neuralNetwork2]

    numberOfNeuralNetworks1Wins = 0
    numberOfNeuralNetworks2Wins = 0
    numberOfDraws = 0

    neuralNetwork1TotalTime = 0.0
    neuralNetwork2TotalTime = 0.0

    for gameNdx in range(args.numberOfGames):
        numberOfPlayedMoves = gameNdx % 2 # Swap the 1st playing neural network et each game
        winner = None
        positionTensor = authority.InitialPosition()
        while winner is None:
            player = playersList[numberOfPlayedMoves % 2]
            playingNeuralNetwork = neuralNetworks[numberOfPlayedMoves % 2]

            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            startTime = time.time()
            moveTensor = AskTheNeuralNetworkToChooseAMove(
                playersList,
                authority,
                playingNeuralNetwork,
                positionTensor,
                depthOfExhaustiveSearch=args.maximumDepthOfSemiExhaustiveSearch,
                numberOfTopMovesToDevelop=args.numberOfTopMovesToDevelop,
                softMaxTemperature=args.softMaxTemperature
            )
            endTime = time.time()
            decisionTime = endTime - startTime
            positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)
            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            if args.displayPositions:
                authority.Display(positionTensor)
                print ("**********************************************")

            if winner is playersList[0] and player is playersList[1]:
                winner = playersList[1]
            elif winner is playersList[1] and player is playersList[1]:
                winner = playersList[0]
            numberOfPlayedMoves += 1
            if player is playersList[0]:
                neuralNetwork1TotalTime += decisionTime
            else:
                neuralNetwork2TotalTime += decisionTime

        if winner is playersList[0]:
            numberOfNeuralNetworks1Wins += 1
        elif winner is playersList[1]:
            numberOfNeuralNetworks2Wins += 1
        elif winner is 'draw':
            numberOfDraws += 1
        else:
            raise NotImplementedError('neuralNetworksTournament.py main(): Unknown winner {}'.format(winner))

        numberOfPlayedMoves = numberOfPlayedMoves - gameNdx % 2 # Subtract 1 for odd game index

    neuralNetwork1WinRate = numberOfNeuralNetworks1Wins / args.numberOfGames
    neuralNetwork2WinRate = numberOfNeuralNetworks2Wins / args.numberOfGames
    drawRate = numberOfDraws / args.numberOfGames

    neuralNetwork1AverageDecisionTime = neuralNetwork1TotalTime/args.numberOfGames
    neuralNetwork2AverageDecisionTime = neuralNetwork2TotalTime / args.numberOfGames

    print ("neuralNetwork1WinRate = {}".format(neuralNetwork1WinRate))
    print ("neuralNetwork2WinRate = {}".format(neuralNetwork2WinRate))
    print ("drawRate = {}".format(drawRate))
    print ("neuralNetwork1AverageDecisionTime = {}".format(neuralNetwork1AverageDecisionTime))
    print ("neuralNetwork2AverageDecisionTime = {}".format(neuralNetwork2AverageDecisionTime))



if __name__ == '__main__':
    main()