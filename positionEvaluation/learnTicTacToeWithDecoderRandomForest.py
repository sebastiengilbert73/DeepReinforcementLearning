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
parser.add_argument('--numberOfSimulations', help='For each starting position, the number of simulations to evaluate the position value. Default: 30', type=int, default=30)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0


def SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions,
                        numberOfPositions):
    selectedPositionsList = []
    while len(selectedPositionsList) < numberOfPositions:
        gamePositionsList, winner = Predictor.SimulateAGame(
            evaluator=None, gameAuthority=authority, startingPosition=None, nextPlayer=None, epsilon=1.0) # With epsilon=1.0, the evaluator will never be called
        if len(gamePositionsList) > minimumNumberOfMovesForInitialPositions:
            if len(gamePositionsList) >= maximumNumberOfMovesForInitialPositions:
                selectedPositionsList.append(gamePositionsList[maximumNumberOfMovesForInitialPositions - 1])
            else:
                selectedPositionsList.append(gamePositionsList[-1])
    return selectedPositionsList

def StartingPositionsTensor(startingPositionsList):
    positionShape = startingPositionsList[0].shape
    startingPositionsTensor = torch.zeros(len(startingPositionsList), positionShape[0], positionShape[1],
                                       positionShape[2], positionShape[3])
    for positionNdx in range(len(startingPositionsList)):
        startingPositionsTensor[positionNdx] = startingPositionsList[positionNdx]
    return startingPositionsTensor

def ExpectedRewardsList(gameAuthority, evaluator, startingPositionsList, numberOfSimulations, nextPlayer, epsilon):
    expectedRewardsList = []
    for positionNdx in range(len(startingPositionsList)):
        startingPosition = startingPositionsList[positionNdx]
        expectedReward = Predictor.ExpectedReward(evaluator, gameAuthority, numberOfSimulations,
                                                  startingPosition=startingPosition,
                                                  nextPlayer=nextPlayer, epsilon=epsilon)
        expectedRewardsList.append(expectedReward)
    return expectedRewardsList

def ExpectedRewardsTensor(expectedRewardsList):
    expectedRewardsTensor = torch.zeros(len(expectedRewardsList))
    for rewardNdx in range(len(expectedRewardsList)):
        expectedRewardsTensor[rewardNdx] = expectedRewardsList[rewardNdx]
    return expectedRewardsTensor

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
        startingPositionsList = SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions,
                                                    maximumNumberOfMovesForInitialPositions,
                                                    args.numberOfPositionsForTraining)
        startingPositionsTensor = StartingPositionsTensor(startingPositionsList)
        logging.info("Evaluating expected reward for each starting position...")
        expectedRewardsList = ExpectedRewardsList(authority, decoderRandomForest, startingPositionsList,
                                                  args.numberOfSimulations, playerList[1], args.epsilon)
        #print ("expectedRewardsList = {}".format(expectedRewardsList))
        expectedRewardsTensor = ExpectedRewardsTensor(expectedRewardsList)

        logging.info("Learning from the examples...")
        decoderRandomForest.LearnFromMinibatch(startingPositionsTensor, expectedRewardsTensor)

        # Test on validation positions
        logging.info("Generating validation positions...")
        validationStartingPositionsList = SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions,
                                                    maximumNumberOfMovesForInitialPositions,
                                                    args.numberOfPositionsForValidation)
        validationStartingPositionsTensor = StartingPositionsTensor(validationStartingPositionsList)
        logging.info("Evaluating expected reward for each validation starting position...")
        validationExpectedRewardsList = ExpectedRewardsList(authority, decoderRandomForest, validationStartingPositionsList,
                                                  args.numberOfSimulations, playerList[1], args.epsilon)
        validationExpectedRewardsTensor = ExpectedRewardsTensor(validationExpectedRewardsList)

        currentValidationPredictionList = decoderRandomForest.Value(validationStartingPositionsTensor)
        currentValidationPredictionTensor = ExpectedRewardsTensor(currentValidationPredictionList)
        validationMSE = torch.nn.functional.mse_loss(currentValidationPredictionTensor, validationExpectedRewardsTensor).item()
        logging.info("validationMSE = {}".format(validationMSE))




if __name__ == '__main__':
    main()