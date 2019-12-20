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
#parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 1', default=1)
#parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.99', type=float, default=0.99)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.1', type=float, default=0.1)
parser.add_argument('--numberOfSimulations', help='For each starting position, the number of simulations to evaluate the position value. Default: 30', type=int, default=30)
parser.add_argument('--maximumNumberOfTrees', help='The maximum number of trees in the forest. Default: 50', type=int, default=50)
parser.add_argument('--treesMaximumDepth', help='The maximum depth of each tree. Default: 6', type=int, default=6)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

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
        autoencoderNet.Load('/home/sebastien/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(2,1,3,3)_[(3,32,1),(3,32,1),(3,32,1)]_20_tictactoeAutoencoder_232.pth')
        decoderRandomForest = Decoder.BuildARandomForestDecoderFromAnAutoencoder(
            autoencoderNet, args.maximumNumberOfTrees, args.treesMaximumDepth)
    decoderRandomForest.SetEvaluationMode('mean')

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
        "epoch,trainingMSE,validationMSE,averageRewardAgainstRandomPlayer,winRate,drawRate,lossRate\n")

    # First game with a random player, before any training
    (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer,
     numberOfDraws) = Predictor.SimulateGamesAgainstARandomPlayer(
        decoderRandomForest, authority, 30
    )
    winRate = numberOfWinsForEvaluator / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
    lossRate = numberOfWinsForRandomPlayer / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
    drawRate = numberOfDraws / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
    logging.info(
        "Against a random player, winRate = {}; drawRate = {}; lossRate = {}".format(winRate, drawRate, lossRate))

    epochLossFile.write(
        '0' + ',' + '-' + ',' + '-' + ',' + str(winRate - lossRate) + ',' + str(
            winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

    for epoch in range(1, args.numberOfEpochs + 1):
        logging.info ("Epoch {}".format(epoch))
        # Generate positions
        minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
        maximumNumberOfMovesForInitialPositions = args.maximumNumberOfMovesForInitialPositions
        logging.info("Generating positions...")
        startingPositionsList = Predictor.SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions,
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

        afterLearningTrainingPredictionsList = decoderRandomForest.Value(startingPositionsTensor)
        afterLearningTrainingPredictionsTensor = ExpectedRewardsTensor(afterLearningTrainingPredictionsList)
        trainingMSE = torch.nn.functional.mse_loss(afterLearningTrainingPredictionsTensor, expectedRewardsTensor).item()
        logging.info("trainingMSE = {}".format(trainingMSE))

        # Test on validation positions
        logging.info("Generating validation positions...")
        validationStartingPositionsList = Predictor.SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions,
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

        # Play against a random player
        (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer,
         numberOfDraws) = Predictor.SimulateGamesAgainstARandomPlayer(
            decoderRandomForest, authority, 30
        )
        winRate = numberOfWinsForEvaluator / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
        lossRate = numberOfWinsForRandomPlayer / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
        drawRate = numberOfDraws / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
        logging.info("Against a random player, winRate = {}; drawRate = {}; lossRate = {}".format(winRate, drawRate, lossRate))

        epochLossFile.write(
            str(epoch) + ',' + str(trainingMSE) + ',' + str(validationMSE) + ',' + str(winRate - lossRate) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')
        filepath = os.path.join(args.outputDirectory, 'tictactoe_' + str(epoch) + '.bin')
        decoderRandomForest.Save(filepath)




if __name__ == '__main__':
    main()