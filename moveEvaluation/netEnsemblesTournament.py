import argparse
import ast
import torch
import time
import random
import time
import ConvolutionStack
import tictactoe
import connect4
import expectedMoveValues
import utilities
import netEnsemble

parser = argparse.ArgumentParser()
parser.add_argument('game', help='The game you want the neural networks to play')
parser.add_argument('neuralNetworks1', help="The list of filepaths for the 1st ensemble")
parser.add_argument('neuralNetworks2', help="The list of filepaths for the 2nd ensemble")
parser.add_argument('--maximumDepthOfSemiExhaustiveSearch', help='The maximum depth of semi exhaustive search. Default: 2', type=int, default=2)
parser.add_argument('--numberOfTopMovesToDevelop', help='For semi-exhaustive search, the number of top moves to develop. Default: 3', type=int, default=3)
parser.add_argument('--softMaxTemperature', help='The softmax temperature, to add inject variance. Default: 0.3', type=float, default=0.3)
parser.add_argument('--numberOfGames', help='The number of games to play. Default: 300', type=int, default=300)
parser.add_argument('--displayPositions', action='store_true', help='Display positions')
args = parser.parse_args()

neuralNetworkFilepathsList1 = ast.literal_eval(args.neuralNetworks1)
neuralNetworkFilepathsList2 = ast.literal_eval(args.neuralNetworks2)


def AskTheEnsembleToChooseAMove(
        playersList,
        authority,
        playingEnsemble,
        positionTensor,
        depthOfExhaustiveSearch,
        numberOfTopMovesToDevelop,
        softMaxTemperature):

    moveValuesTensor, standardDeviationTensor, legalMovesMask = \
    expectedMoveValues.SemiExhaustiveMiniMax(
        playersList,
        authority,
        playingEnsemble,
        positionTensor,
        0.0,
        depthOfExhaustiveSearch,
        1,
        numberOfTopMovesToDevelop
    )

    # Normalize probabilities
    normalizedActionValuesTensor = utilities.NormalizeProbabilities(moveValuesTensor,
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
    print ("netEnsemblesTournament.py main()")

    # Create the game authority
    if args.game == 'tictactoe':
        authority = tictactoe.Authority()
    elif args.game == 'connect4':
        authority = connect4.Authority()
    else:
        raise NotImplementedError("main(): unknown game '{}'".format(args.game))

    playersList = authority.PlayersList()
    #positionTensorShape = authority.PositionTensorShape()
    #moveTensorShape = authority.MoveTensorShape()
    neuralNetworksList1 = []
    for netFilepath in neuralNetworkFilepathsList1:
        neuralNet = ConvolutionStack.Net()
        neuralNet.Load(netFilepath)
        neuralNetworksList1.append(neuralNet)
    netEnsemble1 = netEnsemble.Committee(neuralNetworksList1)

    neuralNetworksList2 = []
    for netFilepath in neuralNetworkFilepathsList2:
        neuralNet = ConvolutionStack.Net()
        neuralNet.Load(netFilepath)
        neuralNetworksList2.append(neuralNet)
    netEnsemble2 = netEnsemble.Committee(neuralNetworksList2)

    netEnsembles = [netEnsemble1, netEnsemble2]

    numberOfNetEnsemble1Wins = 0
    numberOfNetEnsemble2Wins = 0
    numberOfDraws = 0

    netEnsemble1TotalTime = 0.0
    netEnsemble2TotalTime = 0.0

    for gameNdx in range(args.numberOfGames):
        numberOfPlayedMoves = gameNdx % 2 # Swap the 1st playing net ensemble at each game
        winner = None
        positionTensor = authority.InitialPosition()
        while winner is None:
            player = playersList[numberOfPlayedMoves % 2]
            playingNetEnsemble = netEnsembles[numberOfPlayedMoves % 2]

            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            startTime = time.time()
            moveTensor = AskTheEnsembleToChooseAMove(
                playersList,
                authority,
                playingNetEnsemble,
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
                netEnsemble1TotalTime += decisionTime
            else:
                netEnsemble2TotalTime += decisionTime

        if winner is playersList[0]:
            numberOfNetEnsemble1Wins += 1
        elif winner is playersList[1]:
            numberOfNetEnsemble2Wins += 1
        elif winner is 'draw':
            numberOfDraws += 1
        else:
            raise NotImplementedError('neuralNetworksTournament.py main(): Unknown winner {}'.format(winner))

        #numberOfPlayedMoves = numberOfPlayedMoves - gameNdx % 2 # Subtract 1 for odd game index

    netEnsemble1WinRate = numberOfNetEnsemble1Wins / args.numberOfGames
    netEnsemble2WinRate = numberOfNetEnsemble2Wins / args.numberOfGames
    drawRate = numberOfDraws / args.numberOfGames

    netEnsemble1AverageDecisionTime = netEnsemble1TotalTime/args.numberOfGames
    netEnsemble2AverageDecisionTime = netEnsemble2TotalTime / args.numberOfGames

    print ("netEnsemble1WinRate = {}".format(netEnsemble1WinRate))
    print ("netEnsemble2WinRate = {}".format(netEnsemble2WinRate))
    print ("drawRate = {}".format(drawRate))
    print ("netEnsemble1AverageDecisionTime = {}".format(netEnsemble1AverageDecisionTime))
    print ("netEnsemble2AverageDecisionTime = {}".format(netEnsemble2AverageDecisionTime))



if __name__ == '__main__':
    main()