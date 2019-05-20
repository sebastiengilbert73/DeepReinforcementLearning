import argparse
import torch
import time
import policy
import tictactoe
import connect4

parser = argparse.ArgumentParser()
parser.add_argument('game', help='The game you want to play')
parser.add_argument('--neuralNetwork', help="The filepath to the opponent neural network. If None, randomly initialized. Default: None", default=None)
parser.add_argument('networkBodyArchitecture', help="The body architecture. Ex.: '[(3, 16), (3, 16), (3, 16)]'")
parser.add_argument('--opponentPlaysFirst', action='store_true', help='Let the opponent neural network play first')
parser.add_argument('--numberOfGamesForMoveEvaluation', type=int, help='The number of simulated games played by the neural network to evaluate the moves. Default: 31', default=31)
parser.add_argument('--softMaxTemperature', type=float, help='The softMax temperature used by the neural network while simulating the games. Default: 1.0', default=1.0)
parser.add_argument('--displayExpectedMoveValues', action='store_true', help='Display the expected move values, the standard deviations and the legal moves mask')
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The maximum number of moves for exhaustive search. Default: 2', default=2)
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
            positionTensor,
            numberOfGamesForMoveEvaluation,
            softMaxTemperature,
            epsilon,
            displayExpectedMoveValues,
            depthOfExhaustiveSearch):
    moveValuesTensor, standardDeviationTensor, legalMovesMask = policy.PositionExpectedMoveValues(
        playersList,
        authority,
        neuralNetwork,
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

def main():
    print ("gameArena.py main()")

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

    neuralNetwork = policy.NeuralNetwork(positionTensorShape, args.networkBodyArchitecture, moveTensorShape)
    if args.neuralNetwork is not None:
        neuralNetwork.load_state_dict(torch.load(args.neuralNetwork))
    winner = None
    numberOfPlayedMoves = 0
    player = playersList[numberOfPlayedMoves % 2]
    positionTensor = authority.InitialPosition()
    humanPlayerTurn = 0

    if args.opponentPlaysFirst:
        humanPlayerTurn = 1
        moveTensor = AskTheNeuralNetworkToChooseAMove(
            playersList,
            authority,
            neuralNetwork,
            positionTensor,
            args.numberOfGamesForMoveEvaluation,
            args.softMaxTemperature,
            epsilon=0,
            displayExpectedMoveValues=args.displayExpectedMoveValues,
            depthOfExhaustiveSearch=args.depthOfExhaustiveSearch)
        positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)
        numberOfPlayedMoves = 1
        player = playersList[numberOfPlayedMoves % 2]
    authority.Display(positionTensor)

    while winner is None:
        print ("numberOfPlayedMoves % 2 = {}; humanPlayerTurn = {}".format(numberOfPlayedMoves % 2, humanPlayerTurn))
        if numberOfPlayedMoves % 2 == humanPlayerTurn:
            userInput = input ("Your move: ")
            positionTensor, winner = authority.MoveWithString(positionTensor, player, userInput)
            numberOfPlayedMoves += 1
            player = playersList[numberOfPlayedMoves % 2]
            authority.Display(positionTensor)

        else: # Neural network turn
            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            startTime = time.time()
            moveTensor = AskTheNeuralNetworkToChooseAMove(
                playersList,
                authority,
                neuralNetwork,
                positionTensor,
                args.numberOfGamesForMoveEvaluation,
                args.softMaxTemperature,
                epsilon=0,
                displayExpectedMoveValues=args.displayExpectedMoveValues,
                depthOfExhaustiveSearch=args.depthOfExhaustiveSearch)
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