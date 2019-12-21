import argparse
import numpy
import torch
import os
import ast
import tictactoe
import logging
import autoencoder.position # autoencoder
import Decoder
import Predictor

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--startWithAutoencoder', help='The autoencoder whose encoder will be used. Default: None', default=None)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 7', type=int, default=7)
parser.add_argument('--numberOfPositionsForTraining', help='The number of positions for training per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfPositionsForValidation', help='The number of positions for validation per epoch. Default: 128', type=int, default=128)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.2', type=float, default=0.2)
parser.add_argument('--numberOfSimulations', help='For each starting position, the number of simulations to evaluate the position value. Default: 30', type=int, default=30)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--decodingLayerSizesList', help="The list of decoding layer sizes. Default: '[8, 2]'", default='[8, 2]')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
decodingLayerSizesList = ast.literal_eval(args.decodingLayerSizesList)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0

def StartingPositionsTensor(startingPositionsList):
    positionShape = startingPositionsList[0].shape
    startingPositionsTensor = torch.zeros(len(startingPositionsList), positionShape[0], positionShape[1],
                                       positionShape[2], positionShape[3])
    for positionNdx in range(len(startingPositionsList)):
        startingPositionsTensor[positionNdx] = startingPositionsList[positionNdx]
    return startingPositionsTensor

def main():
    logging.info("learnTicTacToeWithDecoderMLP.py main()")

    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        raise NotImplementedError("main(): Start with a neural network is not implemented...")
    else:
        if args.startWithAutoencoder is not None:
            autoencoderNet = autoencoder.position.Net()
            autoencoderNet.Load(args.startWithAutoencoder)

            decoderMLP = Decoder.BuildAnMLPDecoderFromAnAutoencoder(
                autoencoderNet, decodingLayerSizesList)
        else:
            raise NotImplementedError("main(): Starting without an autoencoder is not implemented...")

    # Create the optimizer
    logging.debug(decoderMLP)
    for name, param in decoderMLP.named_parameters():
        if 'decoding' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print("name = {}; param.requires_grad = {}".format(name, param.requires_grad))


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, decoderMLP.parameters()), lr=args.learningRate,
                                 betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.MSELoss()

    # Initial learning rate
    learningRate = args.learningRate


if __name__ == '__main__':
    main()