import argparse
import torch
import os
import ast
import utilities
import expectedMoveValues
import generateMoveStatistics
import tictactoe
import moveEvaluation.ConvolutionStack

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './'", default='./')
parser.add_argument('--proportionOfRandomInitialPositions', help='The proportion of random games in the initial positions. Default: 0.5', type=float, default=0.5)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 20', type=int, default=20)
parser.add_argument('--numberOfInitialPositions', help='The number of initial positions per epoch. Default: 100', type=int, default=100)
parser.add_argument('--numberOfGamesForEvaluation', help='The number of simulated games, for every initial position, for evaluation. default=30', type=int, default=30)
parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.999', type=float, default=0.999)
parser.add_argument('--weightForTheValueLoss', help='The weight to grant to the value loss, with respect to the move probabilities loss. Default: 0.1', type=float, default=0.1)
parser.add_argument('--numberOfStandardDeviationsBelowAverageForValueEstimate', help='When evaluating a position, lower the average value by this number of standard deviations. default: 1.0', type=float, default=1.0)
parser.add_argument('--softMaxTemperatureForSelfPlayEvaluation', help='The softmax temperature when evaluation through self-play. Default: 0.3', type=float, default=0.3)
parser.add_argument('--averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic', help='The dictionary giving the softMax temperature as a function of the average training loss. Default: None (meaning constant softMax temperature)', default=None)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.1', type=float, default=0.1)
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 2', default=2)
parser.add_argument('--chooseHighestProbabilityIfAtLeast', type=float, help='The threshold probability to trigger automatic choice of the highest probability, instead of choosing with roulette. Default: 1.0', default=1.0)
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--numberOfProcesses', help='The number of processes. Default: 4', type=int, default=4)
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


def SoftMaxTemperature(averageTrainingLoss,
                       averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic,
                       initialSoftMaxTemperature):
    keys = list(averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic.keys())
    values = list(averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic.values())
    if averageTrainingLoss > keys[0]:
        return initialSoftMaxTemperature
    foundIndex = None
    for index in range(len(keys) - 1):
        if averageTrainingLoss <= keys[index] and averageTrainingLoss > keys[index + 1]:
            foundIndex = index
    if foundIndex is None: # Last step
        return values[-1]
    else:
        return values[foundIndex]

def HasAlreadyBeenUsed(tensorToCheck, alreadyUsedTensors):
    for existingTensor in alreadyUsedTensors:
        if existingTensor is tensorToCheck:
            return True
    return False

def MinimumNumberOfMovesForInitialPositions(epoch):
    minimumNumberOfMoves = args.maximumNumberOfMovesForInitialPositions - int(epoch/4)
    return max(minimumNumberOfMoves, 0)

def main():

    print ("learnTicTacToe.py main()")

    authority = tictactoe.Authority()
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
    epochLossFile.write("epoch,averageActionValuesTrainingLoss,averageRewardAgainstRandomPlayer,winRate,drawRate,lossRate\n")

    # Save the initial neural network, and write it's score against a random player
    #modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_tictactoe_0.pth")
    #torch.save(neuralNetwork.state_dict(), modelParametersFilename)
    neuralNetwork.Save(args.outputDirectory, 'tictactoe_0')

    (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate, losingGamePositionsListList) = \
        expectedMoveValues.AverageRewardAgainstARandomPlayerKeepLosingGames(
            playerList,
            authority,
            neuralNetwork,
            args.chooseHighestProbabilityIfAtLeast,
            True,
            softMaxTemperature=0.1,
            numberOfGames=300,
            moveChoiceMode='SemiExhaustiveMiniMax',
            numberOfGamesForMoveEvaluation=0,  # ignored by SoftMax
            depthOfExhaustiveSearch=3,
            numberOfTopMovesToDevelop=3
        )
    print ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
        averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))

    epochLossFile.write(
        '0' + ',' + '-' + ',' + str(
            averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

    #bestValidationLoss = sys.float_info.max
    softMaxTemperatureForSelfPlayEvaluation = args.softMaxTemperatureForSelfPlayEvaluation
    if args.averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic is not None:
        averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic = ast.literal_eval(
            args.averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic
        )
    else:
        averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic = None

    losingGamesAgainstRandomPlayerPositionsList = []

    for epoch in range(1, args.numberOfEpochs + 1):
        print ("epoch {}".format(epoch))
        # Set the neural network to training mode
        neuralNetwork.train()

        # Generate positions
        print ("Generating positions...")
        minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
        maximumNumberOfMovesForInitialPositions = args.maximumNumberOfMovesForInitialPositions

        if epoch %4 == -1: #0:
            positionStatisticsList = generateMoveStatistics.GenerateMoveStatisticsMultiprocessing(
                playerList,
                authority,
                neuralNetwork,
                args.proportionOfRandomInitialPositions,
                (minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions),
                args.numberOfInitialPositions,
                args.numberOfGamesForEvaluation,
                softMaxTemperatureForSelfPlayEvaluation,
                args.epsilon,
                args.depthOfExhaustiveSearch,
                args.chooseHighestProbabilityIfAtLeast,
                losingGamesAgainstRandomPlayerPositionsList,
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
                []#losingGamesAgainstRandomPlayerPositionsList
            )
            # (initialPosition, averageValuesTensor, standardDeviationTensor, legalMovesNMask)
        #print ("positionStatisticsList = {}".format(positionStatisticsList))

        #print ("main(): len(positionToMoveProbabilitiesAndValueDic) = {}".format(len(positionToMoveProbabilitiesAndValueDic)))
        #positionsList = list(positionToMoveProbabilitiesAndValueDic.keys())

        actionValuesLossSum = 0.0
        minibatchIndicesList = utilities.MinibatchIndices(len(positionStatisticsList), args.minibatchSize)



        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)
            minibatchPositions = []
            minibatchTargetActionValues = []
            minibatchLegalMovesMasks = []

            for index in minibatchIndicesList[minibatchNdx]:
                #minibatchPositions.append(positionsList[index])
                #if HasAlreadyBeenUsed(positionMoveProbabilityAndValueList[index][0], minibatchPositions):
                #    print ("main(): positionMoveProbabilityAndValueList[index][0] has laready been used")



                #(minibatchMoveProbabilities, value) = \
                #    (positionMoveProbabilityAndValueList[index][1], positionMoveProbabilityAndValueList[index][2])
                    #positionToMoveProbabilitiesAndValueDic[positionsList[index]]
                #if HasAlreadyBeenUsed(positionMoveProbabilityAndValueList[index][1], minibatchTargetMoveProbabilities):
                #    print ("main(): positionMoveProbabilityAndValueList[index][1] has laready been used")
                minibatchPositions.append(positionStatisticsList[index][0])
                averageValueMinusNStdDev = positionStatisticsList[index][1] - \
                                           args.numberOfStandardDeviationsBelowAverageForValueEstimate * positionStatisticsList[index][2]
                legalMovesMask = positionStatisticsList[index][3]
                """averageValueMinusNStdDev = torch.where(positionStatisticsList[index][3] > 0,
                                                averageValueMinusNStdDev,
                                                       positionStatisticsList[index][3].float()) # Get 0 where the mask is 0
                minibatchTargetMoveProbabilities.append(averageValueMinusNStdDev)
                """
                averageValueMinusNStdDev = averageValueMinusNStdDev * legalMovesMask.float()
                minibatchTargetActionValues.append(averageValueMinusNStdDev)

                #if authority.CurrentSum(positionsList[index]) == 1:
                #print ("main(): sum = {}; value = {}".format(authority.CurrentSum(positionsList[index]), value))
                #print ("main(): minibatchMoveProbabilities = \n{}".format(minibatchMoveProbabilities))
                minibatchLegalMovesMasks.append(legalMovesMask)
                #print ("main(): positionStatisticsList[index][0] = {}".format(positionStatisticsList[index][0]))
                #print ("main(): averageValueMinusNStdDev = {}".format(averageValueMinusNStdDev))
                #print ("main(): legalMovesMask = {}".format(legalMovesMask))
                #print ("main(): averageValueMinusNStdDev.max().item() = {}".format(averageValueMinusNStdDev.max().item()))

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
            #print ("outputMoveProbabilitiesTensor.shape = {}".format(outputMoveProbabilitiesTensor.shape))
            #print ("minibatchTargetMoveProbabilitiesTensor.shape = {}".format(minibatchTargetMoveProbabilitiesTensor.shape))
            #print ("outputValuesTensor.shape = {}".format(outputValuesTensor.shape))
            #print ("minibatchTargetValuesTensor.shape = {}".format(minibatchTargetValuesTensor.shape))
            actionValuesLoss = loss(outputActionValuesTensor, minibatchTargetActionValuesTensor)

            try:
                actionValuesLoss.backward()
                # trainingLossSum += minibatchLoss.item()
                actionValuesLossSum += actionValuesLoss.item()

                # Move in the gradient descent direction
                optimizer.step()
            except Exception as exc:
                print ("Caught excetion: {}".format(exc))
                print('X', end='', flush=True)



        averageActionValuesTrainingLoss = actionValuesLossSum / len(minibatchIndicesList)

        print("\nEpoch {}: averageActionValuesTrainingLoss = {}".format(epoch, averageActionValuesTrainingLoss))

        if averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic is not None:
            softMaxTemperatureForSelfPlayEvaluation = SoftMaxTemperature(averageActionValuesTrainingLoss,
                                                                         averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic,
                                                                         args.softMaxTemperatureForSelfPlayEvaluation)


        # Update the learning rates
        learningRate = learningRate * args.learningRateExponentialDecay
        utilities.adjust_lr(optimizer, learningRate)

        # Save the neural network
        #if validationLoss < bestValidationLoss:
        #    bestValidationLoss = validationLoss
        neuralNetwork.Save(args.outputDirectory, 'tictactoe_' + str(epoch))
        #modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_tictactoe_" + str(epoch) + '.pth')
        #torch.save(neuralNetwork.state_dict(), modelParametersFilename)
        if epoch % 20 == -1:
            moveChoiceMode = 'ExpectedMoveValuesThroughSelfPlay'
            numberOfGames = 100
            depthOfExhaustiveSearch = 2
        else:
            moveChoiceMode = 'SemiExhaustiveMiniMax'
            numberOfGames = 300
            depthOfExhaustiveSearch = 3
            numberOfTopMovesToDevelop = 3
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
        print ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
            averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))
        # Collect the positions from losing games
        losingGamesAgainstRandomPlayerPositionsList = []
        for (losingGamePositionsList, firstPlayer) in losingGamePositionsListList:
            for positionNdx in range(len(losingGamePositionsList) - 1):
                if firstPlayer == playerList[0]: # Keep even positions
                    if positionNdx %2 == 0:
                        losingGamesAgainstRandomPlayerPositionsList.append(losingGamePositionsList[positionNdx])
                else: # fistPlayer == playerList[1] -> Keep odd positions
                    if positionNdx %2 == 1:
                        losingGamesAgainstRandomPlayerPositionsList.append(losingGamePositionsList[positionNdx])

        epochLossFile.write(str(epoch) + ',' + str(averageActionValuesTrainingLoss) + ',' + str(averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

        initialPosition = authority.InitialPosition()
        initialPositionOutput = neuralNetwork(initialPosition.unsqueeze(0))
        print ("main(): initialPositionOutput = \n{}".format(initialPositionOutput))

if __name__ == '__main__':
    main()