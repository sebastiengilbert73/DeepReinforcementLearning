import argparse
import torch
import os
import ast
import utilities
import expectedMoveValues
import generateMoveStatistics
import checkers
import moveEvaluation.ConvolutionStack
import logging
import autoencoder.position

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

standardDeviationAlpha = 0.01

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    minimumNumberOfMoves = args.maximumNumberOfMovesForInitialPositions - 40 - int(epoch/3)
    return max(minimumNumberOfMoves, 0)


def main():
    logging.info ("learnCheckers.py main()")

    authority = checkers.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        neuralNetwork = moveEvaluation.ConvolutionStack.Net()
        neuralNetwork.Load(args.startWithNeuralNetwork)
    else:
        """neuralNetwork = moveEvaluation.ConvolutionStack.Net(
            positionTensorShape,
            [(5, 32), (5, 32), (5, 32)],
            moveTensorShape
        )
        """
        autoencoderNet = autoencoder.position.Net()
        autoencoderNet.Load('/home/sebastien/projects/DeepReinforcementLearning/moveEvaluation/autoencoder/outputs/AutoencoderNet_(6,1,8,8)_[(5,16,2),(5,32,2)]_200_checkersAutoencoder_44.pth')
        neuralNetwork = moveEvaluation.ConvolutionStack.BuildAnActionValueDecoderFromAnAutoencoder(
            autoencoderNet, [(16, 1, 2, 2), (8, 1, 4, 4)], (4, 1, 8, 8))
        for name, p in neuralNetwork.named_parameters():
            logging.info ("layer: {}".format(name))
            if "encoding" in name:
                logging.info("Setting p.requires_grad = False")
                p.requires_grad = False
    print ("main(): neuralNetwork = {}".format(neuralNetwork))

    # Create the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, neuralNetwork.parameters()), lr=args.learningRate, betas=(0.5, 0.999))

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
    neuralNetwork.eval()
    (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate, losingGamePositionsListList) = \
        expectedMoveValues.AverageRewardAgainstARandomPlayerKeepLosingGames(
            playerList,
            authority,
            neuralNetwork,
            args.chooseHighestProbabilityIfAtLeast,
            True,
            softMaxTemperature=0.1,
            numberOfGames=11,
            moveChoiceMode='SemiExhaustiveMiniMax',
            numberOfGamesForMoveEvaluation=0,  # ignored by SoftMax
            depthOfExhaustiveSearch=1,
            numberOfTopMovesToDevelop=7
        )
    logging.info ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
        averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))

    epochLossFile.write(
        '0' + ',' + '-' + ',' + str(
            averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

    softMaxTemperatureForSelfPlayEvaluation = args.softMaxTemperatureForSelfPlayEvaluation
    losingGamesAgainstRandomPlayerPositionsList = []

    for epoch in range(1, args.numberOfEpochs + 1):
        logging.info ("Epoch {}".format(epoch))
        # Set the neural network to training mode
        neuralNetwork.train()

        # Generate positions
        minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
        maximumNumberOfMovesForInitialPositions = args.maximumNumberOfMovesForInitialPositions
        logging.info ("Generating positions...")
        if epoch %3 == -1:
            positionStatisticsList = generateMoveStatistics.GenerateMoveStatisticsMultiprocessing(
                playerList,
                authority,
                neuralNetwork,
                args.proportionOfRandomInitialPositions,
                (minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions),
                16, #args.numberOfInitialPositions,
                args.numberOfGamesForEvaluation,
                softMaxTemperatureForSelfPlayEvaluation,
                args.epsilon,
                args.depthOfExhaustiveSearch,
                args.chooseHighestProbabilityIfAtLeast,
                [], #losingGamesAgainstRandomPlayerPositionsList,
                args.numberOfProcesses
            )
        else:
            positionStatisticsList = generateMoveStatistics.GenerateMoveStatisticsWithMiniMax(
                playerList,
                authority,
                neuralNetwork,
                (minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions),
                args.numberOfInitialPositions,
                args.depthOfExhaustiveSearch,
                [],#losingGamesAgainstRandomPlayerPositionsList
            )
        # Add end games
        logging.info("Generating end games...")
        keepNumberOfMovesBeforeEndGame = 3
        numberOfEndGamePositions = 32
        numberOfGamesForEndGameEvaluation = 15
        maximumNumberOfMovesForFullGameSimulation = args.maximumNumberOfMovesForInitialPositions
        maximumNumberOfMovesForEndGameSimulation = 10
        endGamePositionsStatisticsList = generateMoveStatistics.GenerateEndGameStatistics(
            playerList,
            authority,
            neuralNetwork,
            keepNumberOfMovesBeforeEndGame,
            numberOfEndGamePositions,
            numberOfGamesForEndGameEvaluation,
            softMaxTemperatureForSelfPlayEvaluation,
            args.epsilon,
            maximumNumberOfMovesForFullGameSimulation,
            maximumNumberOfMovesForEndGameSimulation,
        )
        #logging.debug("len(positionStatisticsList) = {}; len(endGamePositionsStatisticsList) = {}".format(len(positionStatisticsList), len(endGamePositionsStatisticsList)))
        positionStatisticsList += endGamePositionsStatisticsList
        #logging.debug("After +=: len(positionStatisticsList) = {}".format(len(positionStatisticsList)))

        actionValuesLossSum = 0.0
        minibatchIndicesList = utilities.MinibatchIndices(len(positionStatisticsList), args.minibatchSize)

        logging.info("Going through the minibatch")
        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)
            minibatchPositions = []
            minibatchTargetActionValues = []
            minibatchLegalMovesMasks = []
            for index in minibatchIndicesList[minibatchNdx]:
                #logging.debug("len(positionStatisticsList[{}]) = {}".format(index, len(positionStatisticsList[index])))
                minibatchPositions.append(positionStatisticsList[index][0])
                averageValue = positionStatisticsList[index][1] #- \
                                           #args.numberOfStandardDeviationsBelowAverageForValueEstimate * positionStatisticsList[index][2]
                legalMovesMask = positionStatisticsList[index][3]
                averageValue = averageValue * legalMovesMask.float()
                minibatchTargetActionValues.append(averageValue)
                minibatchLegalMovesMasks.append(legalMovesMask)
            minibatchPositionsTensor = utilities.MinibatchTensor(minibatchPositions)
            minibatchTargetActionValuesTensor = utilities.MinibatchTensor(minibatchTargetActionValues)

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
            # Create a tensor with the list of legal values mask
            minibatchLegalMovesMasksTensor = torch.zeros(outputActionValuesTensor.shape)
            for maskNdx in range(len(minibatchLegalMovesMasks)):
                minibatchLegalMovesMasksTensor[maskNdx] = minibatchLegalMovesMasks[maskNdx]

            standardDeviationOfLegalValues = utilities.StandardDeviationOfLegalValues(outputActionValuesTensor, minibatchLegalMovesMasksTensor)
            logging.debug("standardDeviationOfLegalValues = {}".format(standardDeviationOfLegalValues))
            actionValuesLoss = loss(outputActionValuesTensor, minibatchTargetActionValuesTensor) - standardDeviationAlpha * standardDeviationOfLegalValues

            try:
                actionValuesLoss.backward()
                actionValuesLossSum += actionValuesLoss.item()

                # Move in the gradient descent direction
                optimizer.step()
            except Exception as exc:
                msg = "Caught excetion: {}".format(exc)
                print (msg)
                logging.error(msg)
                print('X', end='', flush=True)

        averageActionValuesTrainingLoss = actionValuesLossSum / len(minibatchIndicesList)
        print(" * ")
        logging.info("Epoch {}: averageActionValuesTrainingLoss = {}".format(epoch, averageActionValuesTrainingLoss))

        # Update the learning rate
        learningRate = learningRate * args.learningRateExponentialDecay
        utilities.adjust_lr(optimizer, learningRate)

        # Save the neural network
        #modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_connect4_" + str(epoch) + '.pth')
        #torch.save(neuralNetwork.state_dict(), modelParametersFilename)
        neuralNetwork.Save(args.outputDirectory, 'checkers_' + str(epoch))
        neuralNetwork.eval()
        if epoch % 200 == -1:
            moveChoiceMode = 'ExpectedMoveValuesThroughSelfPlay'
            numberOfGames = 100
            depthOfExhaustiveSearch = 2
            monitoringSoftMaxTemperature = 0.1
        else:
            moveChoiceMode = 'SemiExhaustiveMiniMax'
            numberOfGames = 11
            depthOfExhaustiveSearch = 1
            numberOfTopMovesToDevelop = 7
        (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate, losingGamePositionsListList) = \
            expectedMoveValues.AverageRewardAgainstARandomPlayerKeepLosingGames(
                playerList,
                authority,
                neuralNetwork,
                args.chooseHighestProbabilityIfAtLeast,
                True,
                softMaxTemperature=softMaxTemperatureForSelfPlayEvaluation,
                numberOfGames=numberOfGames,
                moveChoiceMode=moveChoiceMode,
                numberOfGamesForMoveEvaluation=41,  # ignored by SoftMax
                depthOfExhaustiveSearch=depthOfExhaustiveSearch,
                numberOfTopMovesToDevelop=numberOfTopMovesToDevelop
            )
        logging.info ("averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
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

        """initialPosition = authority.InitialPosition()
        initialPositionOutput = neuralNetwork(initialPosition.unsqueeze(0))
        print("main(): initialPositionOutput = \n{}".format(initialPositionOutput))"""

if __name__ == '__main__':
    main()