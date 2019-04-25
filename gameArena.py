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

args = parser.parse_args()


def AskTheNeuralNetworkToChooseAMove(
            playersList,
            authority,
            neuralNetwork,
            positionTensor,
            numberOfGamesForMoveEvaluation,
            softMaxTemperature,
            epsilon):
    moveValuesTensor, standardDeviationTensor, legalMovesMask = policy.PositionExpectedMoveValues(
        playersList,
        authority,
        neuralNetwork,
        positionTensor,
        numberOfGamesForMoveEvaluation,
        softMaxTemperature,
        epsilon=0
        )
    print ("AskTheNeuralNetworkToChooseAMove():")
    print ("moveValuesTensor = \n{}".format(moveValuesTensor))
    print ("standardDeviationTensor =\n{}".format(standardDeviationTensor))
    print ("legalMovesMask =\n{}".format(legalMovesMask))
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

    positionTensor = authority.InitialPosition()
    if args.opponentPlaysFirst:
        moveTensor = AskTheNeuralNetworkToChooseAMove(
            playersList,
            authority,
            neuralNetwork,
            positionTensor,
            args.numberOfGamesForMoveEvaluation,
            args.softMaxTemperature,
            epsilon=0)
        positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)

    authority.Display(positionTensor)

    while winner is None:
        userInput = input ("Your move: ")

if __name__ == '__main__':
    main()