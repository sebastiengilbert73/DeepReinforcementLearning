import argparse
import torch
import policy
import tictactoe

parser = argparse.ArgumentParser()
parser.add_argument('game', help='The game you want to play')
parser.add_argument('neuralNetwork', help='The filepath to the opponent neural network')
parser.add_argument('networkBodyArchitecture', help="The body architecture. Ex.: '[(3, 16), (3, 16), (3, 16)]'")
parser.add_argument('--opponentPlaysFirst', action='store_true', help='Let the opponent neural network play first')
parser.add_argument('--numberOfGamesForMoveEvaluation', type=int, help='The number of simulated games played by the neural network to evaluate the moves. Default: 31', default=31)
parser.add_argument('--softMaxTemperature', type=float, help='The softMax temperature used by the neural network while simulating the games. Default: 1.0', default=1.0)
parser.add_argument('--displayExpectedMoveValues', action='store_true', help='Display the expected move values, the standard deviations and the legal moves mask')
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
            displayExpectedMoveValues=False):
    moveValuesTensor, standardDeviationTensor, legalMovesMask = policy.PositionExpectedMoveValues(
        playersList,
        authority,
        neuralNetwork,
        positionTensor,
        numberOfGamesForMoveEvaluation,
        softMaxTemperature,
        epsilon
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
    else:
        raise NotImplementedError("main(): unknown game '{}'".format(args.game))

    playersList = authority.PlayersList()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()

    neuralNetwork = policy.NeuralNetwork(positionTensorShape, args.networkBodyArchitecture, moveTensorShape)
    neuralNetwork.load_state_dict(torch.load(args.neuralNetwork))
    winner = None
    numberOfPlayedMoves = 0
    player = playersList[numberOfPlayedMoves % 2]
    positionTensor = authority.InitialPosition()

    if args.opponentPlaysFirst:
        moveTensor = AskTheNeuralNetworkToChooseAMove(
            playersList,
            authority,
            neuralNetwork,
            positionTensor,
            args.numberOfGamesForMoveEvaluation,
            args.softMaxTemperature,
            epsilon=0,
            displayExpectedMoveValues=args.displayExpectedMoveValues)
        positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)
        numberOfPlayedMoves += 1
        player = playersList[numberOfPlayedMoves % 2]
    authority.Display(positionTensor)

    while winner is None:
        userInput = input ("Your move: ")
        positionTensor, winner = authority.MoveWithString(positionTensor, player, userInput)
        numberOfPlayedMoves += 1
        player = playersList[numberOfPlayedMoves % 2]
        authority.Display(positionTensor)

        if winner is None:
            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            moveTensor = AskTheNeuralNetworkToChooseAMove(
                playersList,
                authority,
                neuralNetwork,
                positionTensor,
                args.numberOfGamesForMoveEvaluation,
                args.softMaxTemperature,
                epsilon=0,
                displayExpectedMoveValues=args.displayExpectedMoveValues)
            positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)
            if player is playersList[1]:
                positionTensor = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
            numberOfPlayedMoves += 1
            player = playersList[numberOfPlayedMoves % 2]
            authority.Display(positionTensor)

    if winner == 'draw':
        print ("Draw!")
    else:
        print ("{} won!".format(winner))

if __name__ == '__main__':
    main()