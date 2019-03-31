import argparse
import torch
import os
import ast
import policy
import tictactoe

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
#parser.add_argument('--numberOfStandardDeviationsBelowAverageForValueEstimate', help='When evaluating a position, lower the average value by this number of standard deviations. default: 1.0', type=float, default=1.0)
parser.add_argument('--softMaxTemperatureForSelfPlayEvaluation', help='The softmax temperature when evaluation through self-play. Default: 0.3', type=float, default=0.3)
parser.add_argument('--averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic', help='The dictionary giving the softMax temperature as a function of the average training loss. Default: None (meaning constant softMax temperature)', default=None)
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

def main():

    print ("learnTicTacToe.py main()")

    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    neuralNetwork = policy.NeuralNetwork(positionTensorShape,
                                         '[(3, 16), (3, 16), (3, 16)]',
                                         moveTensorShape)


    # Create the optimizer
    optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=args.learningRate, betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.MSELoss()

    # Initial learning rate
    learningRate = args.learningRate

    # Output monitoring file
    epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    epochLossFile.write("epoch,averageProbTrainingLoss,averageValueTrainingLoss,averageRewardAgainstRandomPlayer,winRate,drawRate,lossRate\n")

    #bestValidationLoss = sys.float_info.max
    softMaxTemperatureForSelfPlayEvaluation = args.softMaxTemperatureForSelfPlayEvaluation
    if args.averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic is not None:
        averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic = ast.literal_eval(
            args.averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic
        )
    else:
        averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic = None

    for epoch in range(1, args.numberOfEpochs + 1):
        print ("epoch {}".format(epoch))
        # Set the neural network to training mode
        neuralNetwork.train()

        # Generate positions
        print ("Generating positions...")
        positionMoveProbabilityAndValueList = policy.GeneratePositionMoveProbabilityAndValue(
            playerList, authority, neuralNetwork,
            args.proportionOfRandomInitialPositions,
            args.maximumNumberOfMovesForInitialPositions,
            args.numberOfInitialPositions,
            args.numberOfGamesForEvaluation,
            softMaxTemperatureForSelfPlayEvaluation
        )

        #print ("main(): len(positionToMoveProbabilitiesAndValueDic) = {}".format(len(positionToMoveProbabilitiesAndValueDic)))
        #positionsList = list(positionToMoveProbabilitiesAndValueDic.keys())

        trainingProbLossSum = 0.0
        trainingValueLossSum = 0.0
        minibatchIndicesList = policy.MinibatchIndices(len(positionMoveProbabilityAndValueList), args.minibatchSize)



        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)
            minibatchPositions = []
            minibatchTargetMoveProbabilities = []
            minibatchTargetValues = []


            for index in minibatchIndicesList[minibatchNdx]:
                #minibatchPositions.append(positionsList[index])
                if HasAlreadyBeenUsed(positionMoveProbabilityAndValueList[index][0], minibatchPositions):
                    print ("main(): positionMoveProbabilityAndValueList[index][0] has laready been used")
                minibatchPositions.append(positionMoveProbabilityAndValueList[index][0])


                #(minibatchMoveProbabilities, value) = \
                #    (positionMoveProbabilityAndValueList[index][1], positionMoveProbabilityAndValueList[index][2])
                    #positionToMoveProbabilitiesAndValueDic[positionsList[index]]
                if HasAlreadyBeenUsed(positionMoveProbabilityAndValueList[index][1], minibatchTargetMoveProbabilities):
                    print ("main(): positionMoveProbabilityAndValueList[index][1] has laready been used")
                minibatchTargetMoveProbabilities.append(positionMoveProbabilityAndValueList[index][1])

                minibatchTargetValues.append(positionMoveProbabilityAndValueList[index][2])
                #if authority.CurrentSum(positionsList[index]) == 1:
                #print ("main(): sum = {}; value = {}".format(authority.CurrentSum(positionsList[index]), value))
                #print ("main(): minibatchMoveProbabilities = \n{}".format(minibatchMoveProbabilities))

            minibatchPositionsTensor = policy.MinibatchTensor(minibatchPositions)
            minibatchTargetMoveProbabilitiesTensor = policy.MinibatchTensor(minibatchTargetMoveProbabilities)
            minibatchTargetValuesTensor = policy.MinibatchValuesTensor(minibatchTargetValues)

            optimizer.zero_grad()

            # Forward pass
            (outputMoveProbabilitiesTensor, outputValuesTensor) = neuralNetwork(minibatchPositionsTensor)

            # Calculate the error and backpropagate
            #print ("outputMoveProbabilitiesTensor.shape = {}".format(outputMoveProbabilitiesTensor.shape))
            #print ("minibatchTargetMoveProbabilitiesTensor.shape = {}".format(minibatchTargetMoveProbabilitiesTensor.shape))
            #print ("outputValuesTensor.shape = {}".format(outputValuesTensor.shape))
            #print ("minibatchTargetValuesTensor.shape = {}".format(minibatchTargetValuesTensor.shape))
            probLoss = (1 - args.weightForTheValueLoss) * loss(outputMoveProbabilitiesTensor, minibatchTargetMoveProbabilitiesTensor)
            valueLoss = args.weightForTheValueLoss * loss(outputValuesTensor, minibatchTargetValuesTensor)
            minibatchLoss = probLoss + valueLoss#(1 - args.weightForTheValueLoss) * loss(outputMoveProbabilitiesTensor, minibatchTargetMoveProbabilitiesTensor) + \
                #args.weightForTheValueLoss * loss(outputValuesTensor, minibatchTargetValuesTensor)
            try:
                minibatchLoss.backward()
                # trainingLossSum += minibatchLoss.item()
                trainingProbLossSum += probLoss.item()
                trainingValueLossSum += valueLoss.item()

                # Move in the gradient descent direction
                optimizer.step()
            except:
                print('X', end='', flush=True)



        averageProbTrainingLoss = trainingProbLossSum / len(minibatchIndicesList)
        averageValueTrainingLoss = trainingValueLossSum / len(minibatchIndicesList)
        print("\nEpoch {}: averageProbTrainingLoss = {}; averageValueTrainingLoss = {}".format(epoch, averageProbTrainingLoss, averageValueTrainingLoss))

        if averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic is not None:
            softMaxTemperatureForSelfPlayEvaluation = SoftMaxTemperature(averageProbTrainingLoss + averageValueTrainingLoss,
                                                                         averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic,
                                                                         args.softMaxTemperatureForSelfPlayEvaluation)


        # Update the learning rates
        learningRate = learningRate * args.learningRateExponentialDecay
        policy.adjust_lr(optimizer, learningRate)

        # Save the neural network
        #if validationLoss < bestValidationLoss:
        #    bestValidationLoss = validationLoss
        modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_tictactoe_" + str(epoch) + '.pth')
        torch.save(neuralNetwork.state_dict(), modelParametersFilename)

        (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate) = \
            policy.AverageRewardAgainstARandomPlayer(
            playerList,
            authority,
            neuralNetwork,
            True,
            0.1,
            300
        )
        print ("main(): averageRewardAgainstRandomPlayer = {}; winRate = {}; drawRate = {}; lossRate = {}".format(
            averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate))

        epochLossFile.write(str(epoch) + ',' + str(averageProbTrainingLoss) + ',' + str(averageValueTrainingLoss) + ',' + str(averageRewardAgainstRandomPlayer) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')


if __name__ == '__main__':
    main()