import argparse
import torch
import os
import ast
import policy
import connect4
import moveEvaluation.ConvolutionStack

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './'", default='./')
parser.add_argument('--proportionOfRandomInitialPositions', help='The proportion of random games in the initial positions. Default: 1.0', type=float, default=1.0)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 20', type=int, default=20)
parser.add_argument('--numberOfInitialPositions', help='The number of initial positions per epoch. Default: 100', type=int, default=100)
parser.add_argument('--numberOfGamesForEvaluation', help='The number of simulated games, for every initial position, for evaluation. default=30', type=int, default=30)
parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.999', type=float, default=0.999)
parser.add_argument('--softMaxTemperatureForSelfPlayEvaluation', help='The softmax temperature when evaluation through self-play. Default: 0.3', type=float, default=0.3)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.1', type=float, default=0.1)
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 1', default=1)
parser.add_argument('--chooseHighestProbabilityIfAtLeast', type=float, help='The threshold probability to trigger automatic choice of the highest probability, instead of choosing with roulette. Default: 1.0', default=1.0)
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


def main():
    print ("learnConnect4.py main()")

    authority = connect4.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    """neuralNetwork = policy.NeuralNetwork(positionTensorShape,
                                         '[(5, 48)]',
                                         moveTensorShape)
    """
    neuralNetwork = moveEvaluation.ConvolutionStack.Net(
        positionTensorShape,
        [(5, 16), (5, 16), (5, 16)],
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
    modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_connect4_0.pth")
    torch.save(neuralNetwork.state_dict(), modelParametersFilename)
    (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate) = \
        policy.AverageRewardAgainstARandomPlayer(
            playerList,
            authority,
            neuralNetwork,
            args.chooseHighestProbabilityIfAtLeast,
            True,
            0.1,
            100,
            moveChoiceMode='SoftMax',
            numberOfGamesForMoveEvaluation=31  # ignored by SoftMax
        )
    print ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
        averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))

    epochLossFile.write(
        '0' + ',' + '-' + ',' + str(
            averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

    softMaxTemperatureForSelfPlayEvaluation = args.softMaxTemperatureForSelfPlayEvaluation
    losingGamesAgainstRandomPlayerPositionsList = []

    for epoch in range(1, args.numberOfEpochs + 1):
        print ("epoch {}".format(epoch))
        # Set the neural network to training mode
        neuralNetwork.train()

        # Generate positions
        print ("Generating positions...")
        positionStatisticsList = policy.GenerateMoveStatistics(
            playerList,
            authority,
            neuralNetwork,
            args.proportionOfRandomInitialPositions,
            args.maximumNumberOfMovesForInitialPositions,
            args.numberOfInitialPositions,
            args.numberOfGamesForEvaluation,
            softMaxTemperatureForSelfPlayEvaluation,
            args.epsilon,
            args.depthOfExhaustiveSearch,
            args.chooseHighestProbabilityIfAtLeast,
            losingGamesAgainstRandomPlayerPositionsList
        )

        actionValuesLossSum = 0.0
        minibatchIndicesList = policy.MinibatchIndices(len(positionStatisticsList), args.minibatchSize)

        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)
            minibatchPositions = []
            minibatchTargetActionValues = []
            minibatchLegalMovesMasks = []
            for index in minibatchIndicesList[minibatchNdx]:
                minibatchPositions.append(positionStatisticsList[index][0])
                averageValue = positionStatisticsList[index][1] #- \
                                           #args.numberOfStandardDeviationsBelowAverageForValueEstimate * positionStatisticsList[index][2]
                legalMovesMask = positionStatisticsList[index][3]
                averageValue = averageValue * legalMovesMask.float()
                minibatchTargetActionValues.append(averageValue)
                minibatchLegalMovesMasks.append(legalMovesMask)
            minibatchPositionsTensor = policy.MinibatchTensor(minibatchPositions)
            minibatchTargetActionValuesTensor = policy.MinibatchTensor(minibatchTargetActionValues)

            optimizer.zero_grad()

            # Forward pass
            outputActionValuesTensor = neuralNetwork(minibatchPositionsTensor)
            # Mask the output action values with the legal moves mask
            for maskNdx in range(len(minibatchLegalMovesMasks)):
                outputActionValues = outputActionValuesTensor[maskNdx].clone()
                legalMovesMask = minibatchLegalMovesMasks[maskNdx]
                maskedOutputActionValues = outputActionValues * legalMovesMask.float()
                outputActionValuesTensor[maskNdx] = maskedOutputActionValues

            # Calculate the error and backpropagate
            actionValuesLoss = loss(outputActionValuesTensor, minibatchTargetActionValuesTensor)

            try:
                actionValuesLoss.backward()
                actionValuesLossSum += actionValuesLoss.item()

                # Move in the gradient descent direction
                optimizer.step()
            except Exception as exc:
                print ("Caught excetion: {}".format(exc))
                print('X', end='', flush=True)

        averageActionValuesTrainingLoss = actionValuesLossSum / len(minibatchIndicesList)
        print("\nEpoch {}: averageActionValuesTrainingLoss = {}".format(epoch, averageActionValuesTrainingLoss))

        # Update the learning rate
        learningRate = learningRate * args.learningRateExponentialDecay
        policy.adjust_lr(optimizer, learningRate)

        # Save the neural network
        modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_connect4_" + str(epoch) + '.pth')
        torch.save(neuralNetwork.state_dict(), modelParametersFilename)
        if epoch % 200 == 0:
            moveChoiceMode = 'ExpectedMoveValuesThroughSelfPlay'
            numberOfGames = 100
            depthOfExhaustiveSearch = 2
        else:
            moveChoiceMode = 'SoftMax'
            numberOfGames = 100
            depthOfExhaustiveSearch = 1
        (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate, losingGamePositionsListList) = \
            policy.AverageRewardAgainstARandomPlayerKeepLosingGames(
            playerList,
            authority,
            neuralNetwork,
            args.chooseHighestProbabilityIfAtLeast,
            True,
            softMaxTemperature=softMaxTemperatureForSelfPlayEvaluation,
            numberOfGames=numberOfGames,
            moveChoiceMode=moveChoiceMode,
            numberOfGamesForMoveEvaluation=41,  # ignored by SoftMax
            depthOfExhaustiveSearch=depthOfExhaustiveSearch
        )
        print ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
            averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))

        # Collect the positions from losing games
        losingGamesAgainstRandomPlayerPositionsList = []
        for (losingGamePositionsList, firstPlayer) in losingGamePositionsListList:
            for positionNdx in range(len(losingGamePositionsList) - 1):
                if firstPlayer == playerList[0]:  # Keep even positions
                    if positionNdx % 2 == 0:
                        losingGamesAgainstRandomPlayerPositionsList.append(losingGamePositionsList[positionNdx])
                else:  # fistPlayer == playerList[1] -> Keep odd positions
                    if positionNdx % 2 == 1:
                        losingGamesAgainstRandomPlayerPositionsList.append(losingGamePositionsList[positionNdx])

        epochLossFile.write(str(epoch) + ',' + str(averageActionValuesTrainingLoss) + ',' + str(
            averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')


if __name__ == '__main__':
    main()