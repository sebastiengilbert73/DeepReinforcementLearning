import argparse
import torch
import os
import ast
import utilities
import expectedMoveValues
import generateMoveStatistics
import checkers
import moveEvaluation.ConvolutionStack

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './'", default='./')
parser.add_argument('--proportionOfRandomInitialPositions', help='The proportion of random games in the initial positions. Default: 0.5', type=float, default=0.5)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 40', type=int, default=40)
parser.add_argument('--numberOfInitialPositions', help='The number of initial positions per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfGamesForEvaluation', help='The number of simulated games, for every initial position, for evaluation. default=31', type=int, default=31)
parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.99', type=float, default=0.99)
parser.add_argument('--softMaxTemperatureForSelfPlayEvaluation', help='The softmax temperature when evaluation through self-play. Default: 0.3', type=float, default=0.3)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.1', type=float, default=0.1)
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 1', default=1)
parser.add_argument('--chooseHighestProbabilityIfAtLeast', type=float, help='The threshold probability to trigger automatic choice of the highest probability, instead of choosing with roulette. Default: 1.0', default=1.0)
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--numberOfProcesses', help='The number of processes. Default: 4', type=int, default=4)
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


def MinimumNumberOfMovesForInitialPositions(epoch):
    minimumNumberOfMoves = args.maximumNumberOfMovesForInitialPositions - 20 - int(epoch/1)
    return max(minimumNumberOfMoves, 0)


def main():
    print ("learnCheckers.py main()")

    authority = checkers.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        neuralNetwork = moveEvaluation.ConvolutionStack.Net()
        neuralNetwork.Load(args.startWithNeuralNetwork)
    else:
        neuralNetwork = moveEvaluation.ConvolutionStack.Net(
            positionTensorShape,
            [(3, 32), (3, 32), (3, 32)],
            moveTensorShape
        )

    # Create the optimizer
    optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=args.learningRate, betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.MSELoss()

    # Initial learning rate
    learningRate = args.learningRate

    # Output monitoring file
    epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    epochLossFile.write(
        "epoch,averageActionValuesTrainingLoss,averageRewardAgainstRandomPlayer,winRate,drawRate,lossRate\n")

    # Save the initial neural network, and write it's score against a random player
    neuralNetwork.Save(args.outputDirectory, 'checkers_0')
    (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate, losingGamePositionsListList) = \
        expectedMoveValues.AverageRewardAgainstARandomPlayerKeepLosingGames(
            playerList,
            authority,
            neuralNetwork,
            args.chooseHighestProbabilityIfAtLeast,
            True,
            softMaxTemperature=0.1,
            numberOfGames=100,
            moveChoiceMode='SemiExhaustiveMiniMax',
            numberOfGamesForMoveEvaluation=0,  # ignored by SoftMax
            depthOfExhaustiveSearch=2,
            numberOfTopMovesToDevelop=3
        )
    print ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
        averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))

    epochLossFile.write(
        '0' + ',' + '-' + ',' + str(
            averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

    softMaxTemperatureForSelfPlayEvaluation = args.softMaxTemperatureForSelfPlayEvaluation
    losingGamesAgainstRandomPlayerPositionsList = []

if __name__ == '__main__':
    main()