import argparse
import ast
import torch
import time
import expectedMoveValues
import generateMoveStatistics
import utilities
import moveEvaluation.ConvolutionStack
import tictactoe
import connect4
import checkers
import moveEvaluation.netEnsemble
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('game', help='The game you want to play')
parser.add_argument('--neuralNetwork', help="The filepath to the opponent neural network. If None, randomly initialized. Default: None", default=None)
parser.add_argument('networkBodyArchitecture', help="The body architecture. Ex.: '[(3, 16), (3, 16), (3, 16)]'")
parser.add_argument('--opponentPlaysFirst', action='store_true', help='Let the opponent neural network play first')
parser.add_argument('--numberOfGamesForMoveEvaluation', type=int, help='The number of simulated games played by the neural network to evaluate the moves. Default: 31', default=31)
parser.add_argument('--softMaxTemperature', type=float, help='The softMax temperature used by the neural network while simulating the games. Default: 1.0', default=1.0)
parser.add_argument('--displayExpectedMoveValues', action='store_true', help='Display the expected move values, the standard deviations and the legal moves mask')
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The maximum number of moves for exhaustive search. Default: 2', default=2)
parser.add_argument('--chooseHighestProbabilityIfAtLeast', type=float, help='The threshold probability to trigger automatic choice of the highest probability, instead of choosing with roulette. Default: 1.0', default=1.0)
parser.add_argument('--numberOfTopMovesToDevelop', type=int, help='For SemiExhaustiveMinimax, the number of top moves to develop. Default: 3', default=3)
args = parser.parse_args()


def DisplayExpectedMoveValues(moveValuesTensor, standardDeviationTensor, legalMovesMask, chosenMoveTensor):
    print ("moveValuesTensor = \n{}".format(moveValuesTensor))
    print ("standardDeviationTensor =\n{}".format(standardDeviationTensor))
    print ("legalMovesMask =\n{}".format(legalMovesMask))
    print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))

def AskTheNeuralNetworkToChooseAMove(
            playersList,
            authority,
            neuralNetwork,
            chooseHighestProbabilityIfAtLeast,
            positionTensor,
            numberOfGamesForMoveEvaluation,
            softMaxTemperature,
            epsilon,
            displayExpectedMoveValues,
            depthOfExhaustiveSearch):
    moveValuesTensor, standardDeviationTensor, legalMovesMask = expectedMoveValues.PositionExpectedMoveValues(
        playersList,
        authority,
        neuralNetwork,
        chooseHighestProbabilityIfAtLeast,
        positionTensor,
        numberOfGamesForMoveEvaluation,
        softMaxTemperature,
        epsilon,
        depthOfExhaustiveSearch
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

    if displayExpectedMoveValues:
        DisplayExpectedMoveValues(moveValuesTensor, standardDeviationTensor, legalMovesMask, chosenMoveTensor)

    return chosenMoveTensor

def SemiExhaustiveMinimaxHighestValue(
        playerList,
        authority,
        neuralNetwork,
        positionTensor,
        epsilon,
        maximumDepthOfSemiExhaustiveSearch,
        numberOfTopMovesToDevelop,
        displayExpectedMoveValues,
    ):
    (moveValuesTensor, standardDeviationTensor, legalMovesMask) = expectedMoveValues.SemiExhaustiveMiniMax(
        playerList,
        authority,
        neuralNetwork,
        positionTensor,
        epsilon,
        maximumDepthOfSemiExhaustiveSearch,
        1,
        numberOfTopMovesToDevelop
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

    if displayExpectedMoveValues:
        DisplayExpectedMoveValues(moveValuesTensor, standardDeviationTensor, legalMovesMask, chosenMoveTensor)

    return chosenMoveTensor

def main():
    print ("gameArena.py main()")

    # Create the game authority
    if args.game == 'tictactoe':
        authority = tictactoe.Authority()
    elif args.game == 'connect4':
        authority = connect4.Authority()
    elif args.game == 'checkers':
        authority = checkers.Authority()
    else:
        raise NotImplementedError("main(): unknown game '{}'".format(args.game))

    playersList = authority.PlayersList()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()

    #if type(ast.literal_eval(args.neuralNetwork)) is list: # Neural networks ensemble
    if args.neuralNetwork is not None and args.neuralNetwork.startswith('[') and args.neuralNetwork.endswith(']'): # List => neural networks ensemble
        committeeMembersList = []
        for neuralNetworkFilepath in ast.literal_eval(args.neuralNetwork):
            committeeMember = moveEvaluation.ConvolutionStack.Net()
            committeeMember.Load(neuralNetworkFilepath)
            committeeMembersList.append(committeeMember)
        neuralNetwork = moveEvaluation.netEnsemble.Committee(committeeMembersList)
    else: # Single neural network
        neuralNetwork = moveEvaluation.ConvolutionStack.Net(positionTensorShape,
                                                        ast.literal_eval(args.networkBodyArchitecture),
                                                        moveTensorShape)
        if args.neuralNetwork is not None:
            neuralNetwork.Load(args.neuralNetwork)

    winner = None
    numberOfPlayedMoves = 0
    player = playersList[numberOfPlayedMoves % 2]
    positionTensor = authority.InitialPosition()
    humanPlayerTurn = 0

    if args.opponentPlaysFirst:
        humanPlayerTurn = 1
        """moveTensor = AskTheNeuralNetworkToChooseAMove(
            playersList,
            authority,
            neuralNetwork,
            args.chooseHighestProbabilityIfAtLeast,
            positionTensor,
            args.numberOfGamesForMoveEvaluation,
            args.softMaxTemperature,
            epsilon=0,
            displayExpectedMoveValues=args.displayExpectedMoveValues,
            depthOfExhaustiveSearch=args.depthOfExhaustiveSearch)
        """
        moveTensor = SemiExhaustiveMinimaxHighestValue(
            playersList,
            authority,
            neuralNetwork,
            positionTensor,
            epsilon=0,
            maximumDepthOfSemiExhaustiveSearch=args.depthOfExhaustiveSearch,
            numberOfTopMovesToDevelop=args.numberOfTopMovesToDevelop,
            displayExpectedMoveValues=args.displayExpectedMoveValues,
        )

        positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)
        numberOfPlayedMoves = 1
        player = playersList[numberOfPlayedMoves % 2]
    authority.Display(positionTensor)

    while winner is None:
        print ("numberOfPlayedMoves % 2 = {}; humanPlayerTurn = {}".format(numberOfPlayedMoves % 2, humanPlayerTurn))
        if numberOfPlayedMoves % 2 == humanPlayerTurn:
            inputIsLegal = False
            while not inputIsLegal:
                try:
                    userInput = input("Your move ('?' to get the legal moves mask, 'positionTensor' to get the position tensor): ")
                    if userInput == "?":
                        legalMovesMask = authority.LegalMovesMask(positionTensor, player)
                        print ("legalMovesMask = \n{}".format(legalMovesMask))
                        inputIsLegal = False
                    elif userInput == "positionTensor":
                        print ("positionTensor = \n{}".format(positionTensor))
                    else:
                        positionTensor, winner = authority.MoveWithString(positionTensor, player, userInput)
                        inputIsLegal = True
                except ValueError as e:
                    print ("Caught exception '{}'.\nTry again".format(e))
            numberOfPlayedMoves += 1
            player = playersList[numberOfPlayedMoves % 2]
            authority.Display(positionTensor)

        else: # Neural network turn
            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            startTime = time.time()
            """moveTensor = AskTheNeuralNetworkToChooseAMove(
                playersList,
                authority,
                neuralNetwork,
                args.chooseHighestProbabilityIfAtLeast,
                positionTensor,
                args.numberOfGamesForMoveEvaluation,
                args.softMaxTemperature,
                epsilon=0,
                displayExpectedMoveValues=args.displayExpectedMoveValues,
                depthOfExhaustiveSearch=args.depthOfExhaustiveSearch)
            """
            moveTensor = SemiExhaustiveMinimaxHighestValue(
                playersList,
                authority,
                neuralNetwork,
                positionTensor,
                epsilon=0,
                maximumDepthOfSemiExhaustiveSearch=args.depthOfExhaustiveSearch,
                numberOfTopMovesToDevelop=args.numberOfTopMovesToDevelop,
                displayExpectedMoveValues=args.displayExpectedMoveValues,
            )
            endTime = time.time()
            decisionTime = endTime - startTime
            print ("decisionTime = {}".format(decisionTime))
            positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)
            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            if winner is playersList[0] and player is playersList[1]:
                winner = playersList[1]
            numberOfPlayedMoves += 1
            player = playersList[numberOfPlayedMoves % 2]
            authority.Display(positionTensor)

    if winner == 'draw':
        print ("Draw!")
    else:
        print ("{} won!".format(winner))

if __name__ == '__main__':
    main()