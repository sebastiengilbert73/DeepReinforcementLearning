import argparse
import numpy
import torch
import os
import ast
import tictactoe
import logging
import autoencoder.position # autoencoder
import Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
#parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
#parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 7', type=int, default=7)
parser.add_argument('--numberOfPositionsForTraining', help='The number of positions for training per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfPositionsForValidation', help='The number of positions for validation per epoch. Default: 128', type=int, default=128)
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 1', default=1)
parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.99', type=float, default=0.99)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.1', type=float, default=0.1)


args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0



def main():
    logging.info("learnTicTacToeWithDecoderRandomForest.py main()")

    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        raise NotImplementedError("main(): Start with a neural network is not implemented...")
    else:
        autoencoderNet = autoencoder.position.Net()
        autoencoderNet.Load('/home/segilber/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(2,1,3,3)_[(3,16,1),(3,32,1)]_20_tictactoeAutoencoder_1000.pth')
        decoderRandomForest = Decoder.BuildARandomForestDecoderFromAnAutoencoder(
            autoencoderNet, maximumNumberOfTrees=5, treesMaximumDepth=6)

    print ("main(): decoderRandomForest.encodingBodyStructureSeq = {}".format(decoderRandomForest.encodingBodyStructureSeq))

    """# Create the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, neuralNetwork.parameters()), lr=args.learningRate,
                                 betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.MSELoss()

    # Initial learning rate
    learningRate = args.learningRate
    """

    # Output monitoring file
    epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    epochLossFile.write(
        "epoch,averageTrainError,averageValidationError,averageRewardAgainstRandomPlayer,winRate,drawRate,lossRate\n")

    for epoch in range(1, args.numberOfEpochs + 1):
        logging.info ("Epoch {}".format(epoch))
        # Generate positions
        minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
        maximumNumberOfMovesForInitialPositions = args.maximumNumberOfMovesForInitialPositions
        logging.info("Generating positions...")


if __name__ == '__main__':
    main()