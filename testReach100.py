import argparse
import torch
import policy
import reach100

parser = argparse.ArgumentParser()
parser.add_argument('NeuralNetworkFilepath', help='The neural network')
parser.add_argument('--neuralNetworkStarts', action='store_true', help='Let the neural network start')
args = parser.parse_args()

authority = reach100.Authority()
positionTensorShape = authority.PositionTensorShape()
moveTensorShape = authority.MoveTensorShape()
playersList = authority.PlayersList()
neuralNetwork = policy.NeuralNetwork(
                                    positionTensorShape,
                                    '[(7, 1, 1, 32), (7, 1, 1, 30)]',
                                    moveTensorShape)

neuralNetwork.load_state_dict(torch.load(args.NeuralNetworkFilepath, map_location=lambda storage, location: storage))


def main():
    print ("testReach100.py main()")
    position = authority.InitialPosition()
    moveNdx = 0
    winner = None
    if not args.neuralNetworkStarts: # Ask for a number
        inputNbr = int(input ("Sum = {}. Choose a number [1, 10]: ".format(authority.CurrentSum(position))) )
        position, winner = authority.MoveWithInteger(position, playersList[0], inputNbr)
        moveNdx += 1
        # Swap the players
        position = authority.SwapPositions(position, playersList[0], playersList[1])

    while winner is None:
        # Ask the neural network to choose a move
        player = playersList[moveNdx % 2]
        print ("Sum = {}".format(authority.CurrentSum(position)))
        moveProbabilitiesTensor, value = neuralNetwork(position.unsqueeze(0))
        print ("moveProbabilitiesTensor = \n{}".format(moveProbabilitiesTensor))
        print ("value = {}".format(value))
        chosenMoveTensor = neuralNetwork.ChooseAMove(position,
                                                     player,
                                                     authority,
                                                     preApplySoftMax=True, softMaxTemperature=1.0)
        position, winner = authority.Move(position, player, chosenMoveTensor)
        moveNdx += 1
        # Swap the players
        position = authority.SwapPositions(position, playersList[0], playersList[1])
        print ("The neural network choose {}".format(authority.PlayedValue(chosenMoveTensor)))
        if winner is None:
            player = playersList[moveNdx % 2]
            inputNbr = int(input("Sum = {}. Choose a number [1, 10]: ".format(authority.CurrentSum(position))))
            position, winner = authority.MoveWithInteger(position, playersList[0], inputNbr)
            moveNdx += 1
            # Swap the players
            position = authority.SwapPositions(position, playersList[0], playersList[1])

    print ("winner = {}".format(winner))


if __name__ == '__main__':
    main()