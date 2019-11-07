import argparse
import numpy
import torch
import os
import ast
import utilities
import expectedMoveValues
import generateMoveStatistics
import checkers
import logging
import position # autoencoder


parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './output'", default='./output')
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--maximumNumberOfMovesForPositions', help='The maximum number of moves in the ipositions. Default: 120', type=int, default=120)
parser.add_argument('--numberOfPositionsForTraining', help='The number of positions for training per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfPositionsForValidation', help='The number of positions for validation per epoch. Default: 128', type=int, default=128)
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 1', default=1)
parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.99', type=float, default=0.99)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0

def GenerateRandomPositions(
        numberOfPositions,
        playerList,
        authority,
        maximumNumberOfMoves
        ):
    positionsList = []
    for positionNdx in range(numberOfPositions):
        gamePositionsList, winner = expectedMoveValues.SimulateAGame(
            playerList,
            authority=authority,
            neuralNetwork=None,
            softMaxTemperatureForSelfPlayEvaluation=1.0,  # Will be ignored
            epsilon=0.1,
            maximumNumberOfMoves=maximumNumberOfMoves,
            startingPosition=None,
            opponentPlaysRandomly=True
        )
        positionsList.append(gamePositionsList[-1])  # Keep the last position
    return positionsList

def main():
    print ("learnCheckersAutoencoder.py main()")

    authority = checkers.Authority()
    positionTensorShape = authority.PositionTensorShape()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        neuralNetwork = position.Net()
        neuralNetwork.Load(args.startWithNeuralNetwork)
        for name, p in neuralNetwork.named_parameters():
            logging.info ("layer: {}".format(name))
            if "layer_0" in name or "layer_1" in name:
                logging.info("Setting p.requires_grad = False")
                p.requires_grad = False

    else:
        neuralNetwork = position.Net(
            positionTensorShape,
            bodyStructure=[(3, 16, 2), (3, 32, 2), (3, 64, 2)],#, (5, 16), (5, 16)],
            numberOfLatentVariables=400
        )

    # Create the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, neuralNetwork.parameters()), lr=args.learningRate, betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.BCEWithLogitsLoss()# torch.nn.MSELoss()

    # Initial learning rate
    learningRate = args.learningRate

    # Output monitoring file
    epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    epochLossFile.write("epoch,trainingLoss,validationLoss,errorRate\n")

    for epoch in range(1, args.numberOfEpochs + 1):
        logging.info ("Epoch {}".format(epoch))
        # Set the neural network to training mode
        neuralNetwork.train()

        # Generate positions
        minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
        maximumNumberOfMovesForPositions = numpy.random.randint(args.maximumNumberOfMovesForPositions) # Between 0 and args.maximumNumberOfMovesForPositions
        logging.info("Generating {} training random positions".format(args.numberOfPositionsForTraining))
        trainingPositionsList = GenerateRandomPositions(args.numberOfPositionsForTraining, playerList, authority, maximumNumberOfMovesForPositions)
        logging.info("Generating {} validation random positions".format(args.numberOfPositionsForValidation))
        validationPositionsList = GenerateRandomPositions(args.numberOfPositionsForValidation, playerList, authority, maximumNumberOfMovesForPositions)

        trainingLossSum = 0.0
        minibatchIndicesList = utilities.MinibatchIndices(len(trainingPositionsList), args.minibatchSize)

        logging.info("Going through the minibatch")
        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)

            minibatchPositions = []
            for index in minibatchIndicesList[minibatchNdx]:
                #logging.debug("len(positionStatisticsList[{}]) = {}".format(index, len(positionStatisticsList[index])))
                minibatchPositions.append(trainingPositionsList[index])
            minibatchPositionsTensor = utilities.MinibatchTensor(minibatchPositions)
            minibatchTargetPositionsTensor = utilities.MinibatchTensor(minibatchPositions) # Autoencoder => target output = input

            optimizer.zero_grad()

            # Forward pass
            outputTensor = neuralNetwork(minibatchPositionsTensor)

            # Calculate the error and backpropagate
            trainingLoss = loss(outputTensor, minibatchTargetPositionsTensor)
            #logging.debug("trainingLoss.item() = {}".format(trainingLoss.item()))

            trainingLoss.backward()
            trainingLossSum += trainingLoss.item()

            # Move in the gradient descent direction
            optimizer.step()

        averageTrainingLoss = trainingLossSum / len(minibatchIndicesList)

        # Compute the validation loss
        neuralNetwork.eval()
        validationPositionsTensor = utilities.MinibatchTensor(validationPositionsList)
        validationOutputTensor = neuralNetwork(validationPositionsTensor)
        validationLoss = loss(validationOutputTensor, validationPositionsTensor).item() # Autoencoder => target output = input

        # Compare the output tensor converted to one-hot with the target
        oneHotValidationOutputTensor = position.ConvertToOneHotPositionTensor(validationOutputTensor)
        numberOfErrors = torch.nonzero(validationPositionsTensor.long() - oneHotValidationOutputTensor).shape[0]
        errorRate = numberOfErrors/max(torch.nonzero(validationPositionsTensor).shape[0], 0)

        print(" * ")
        logging.info("Epoch {}: averageTrainingLoss = {}\tvalidationLoss = {}\terrorRate = {}".format(epoch, averageTrainingLoss, validationLoss, errorRate))
        epochLossFile.write(str(epoch) + ',' + str(averageTrainingLoss) + ',' + str(validationLoss) + ',' + str(errorRate) + '\n')

        # Save the neural network
        neuralNetwork.Save(args.outputDirectory, 'checkersAutoencoder_' + str(epoch))

        # Update the learning rate
        learningRate = learningRate * args.learningRateExponentialDecay
        utilities.adjust_lr(optimizer, learningRate)

if __name__ == '__main__':
    main()