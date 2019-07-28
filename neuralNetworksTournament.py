import argparse
import ast
import torch
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
args = parser.parse_args()


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

if __name__ == '__main__':
    main()