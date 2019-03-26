import argparse
import torch
import policy
import reachX
import random
import time
import multiprocessing
import os
import sys
import numpy
import ast

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
parser.add_argument('--targetValue', help='The target value. Default: 12', type=int, default=12)
parser.add_argument('--maximumPlayedValue', help='The maximum played value. Default: 10', type=int, default=10)
parser.add_argument('--softMaxTemperatureForSelfPlayEvaluation', help='The softmax temperature when evaluation through self-play. Default: 0.3', type=float, default=0.3)
parser.add_argument('--averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic', help='The dictionary giving the softMax temperature as a function of the average training loss. Default: None (meaning constant softMax temperature)', default=None)
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

playerList = ['Player1', 'Player2']

"""def MinibatchIndices(numberOfSamples, minibatchSize):
	shuffledList = numpy.arange(numberOfSamples)
	numpy.random.shuffle(shuffledList)
	minibatchesIndicesList = []
	numberOfWholeLists = int(numberOfSamples / minibatchSize)
	for wholeListNdx in range(numberOfWholeLists):
		minibatchIndices = shuffledList[ wholeListNdx * minibatchSize : (wholeListNdx + 1) * minibatchSize ]
		minibatchesIndicesList.append(minibatchIndices)
	# Add the last incomplete minibatch
	if numberOfWholeLists * minibatchSize < numberOfSamples:
		lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
		minibatchesIndicesList.append(lastMinibatchIndices)
	return minibatchesIndicesList
"""

"""def MinibatchTensor(positionsList):
    if len(positionsList) == 0:
        raise ArgumentException("MinibatchTensor(): Empty list of positions")
    positionShape = positionsList[0].shape
    minibatchTensor = torch.zeros(len(positionsList), positionShape[0],
                                  positionShape[1], positionShape[2], positionShape[3]) # NCDHW
    for n in range(len(positionsList)):
        minibatchTensor[n] = positionsList[n]
    return minibatchTensor


def MinibatchValuesTensor(valuesList):
    if len(valuesList) == 0:
        raise ArgumentException("MinibatchValuesTensor(): Empty list of values")
    valuesTensor = torch.zeros(len (valuesList))
    for n in range (len (valuesList)):
        valuesTensor[n] = valuesList[n]
    return valuesTensor

def adjust_lr(optimizer, desiredLearningRate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = desiredLearningRate
"""

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







def main():

    print ("learnReachX.py main()")

    authority = reachX.Authority(args.targetValue, args.maximumPlayedValue)
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()

    neuralNetwork = policy.NeuralNetwork(positionTensorShape,
                                         '[(15, 1, 1, 16), (15, 1, 1, 16), (15, 1, 1, 16)]',
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
    epochLossFile.write("epoch,averageTrainingLoss,averageRewardAgainstRandomPlayer\n")

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
        positionToMoveProbabilitiesAndValueDic = policy.GeneratePositionToMoveProbabilityAndValueDic(
            playerList, authority, neuralNetwork,
            args.proportionOfRandomInitialPositions,
            args.maximumNumberOfMovesForInitialPositions,
            args.numberOfInitialPositions,
            args.numberOfGamesForEvaluation,
            softMaxTemperatureForSelfPlayEvaluation
        )

        #print ("main(): len(positionToMoveProbabilitiesAndValueDic) = {}".format(len(positionToMoveProbabilitiesAndValueDic)))
        positionsList = list(positionToMoveProbabilitiesAndValueDic.keys())

        trainingLossSum = 0.0
        minibatchIndicesList = policy.MinibatchIndices(len(positionsList), args.minibatchSize)

        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)
            minibatchPositions = []
            minibatchTargetMoveProbabilities = []
            minibatchTargetValues = []
            for index in minibatchIndicesList[minibatchNdx]:
                minibatchPositions.append(positionsList[index])
                (minibatchMoveProbabilities, value) = \
                    positionToMoveProbabilitiesAndValueDic[positionsList[index]]
                minibatchTargetMoveProbabilities.append(minibatchMoveProbabilities)
                minibatchTargetValues.append(value)
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
            minibatchLoss = (1 - args.weightForTheValueLoss) * loss(outputMoveProbabilitiesTensor, minibatchTargetMoveProbabilitiesTensor) + \
                args.weightForTheValueLoss * loss(outputValuesTensor, minibatchTargetValuesTensor)
            minibatchLoss.backward()
            trainingLossSum += minibatchLoss.item()

            # Move in the gradient descent direction
            optimizer.step()

        averageTrainingLoss = trainingLossSum / len(minibatchIndicesList)
        print("\nEpoch {}: averageTrainingLoss = {}".format(epoch, averageTrainingLoss))

        if averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic is not None:
            softMaxTemperatureForSelfPlayEvaluation = SoftMaxTemperature(averageTrainingLoss,
                                                                         averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic,
                                                                         args.softMaxTemperatureForSelfPlayEvaluation)


        # Update the learning rates
        learningRate = learningRate * args.learningRateExponentialDecay
        policy.adjust_lr(optimizer, learningRate)

        # Save the neural network
        #if validationLoss < bestValidationLoss:
        #    bestValidationLoss = validationLoss
        modelParametersFilename = os.path.join(args.outputDirectory, "neuralNet_" + str(epoch) + '.pth')
        torch.save(neuralNetwork.state_dict(), modelParametersFilename)

        averageRewardAgainstRandomPlayer = policy.AverageRewardAgainstARandomPlayer(
            playerList,
            authority,
            neuralNetwork,
            True,
            0.1,
            300
        )
        print ("main(): averageRewardAgainstRandomPlayer = {}".format(averageRewardAgainstRandomPlayer))

        epochLossFile.write(str(epoch) + ',' + str(averageTrainingLoss) + ',' + str(averageRewardAgainstRandomPlayer) + '\n')


if __name__ == '__main__':
    main()