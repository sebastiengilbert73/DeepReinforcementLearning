import argparse
import numpy
import torch
import os
import ast
import utilities
import expectedMoveValues
import generateMoveStatistics
import tictactoe
import logging
import position # autoencoder


parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--maximumNumberOfMovesForPositions', help='The maximum number of moves in the positions. Default: 9', type=int, default=9)
parser.add_argument('--numberOfPositionsForTraining', help='The number of positions for training per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfPositionsForValidation', help='The number of positions for validation per epoch. Default: 128', type=int, default=128)
parser.add_argument('--depthOfExhaustiveSearch', type=int, help='The depth of exhaustive search, when generating move statitics. Default: 1', default=1)
parser.add_argument('--learningRateExponentialDecay', help='The learning rate exponential decay. Default: 0.99', type=float, default=0.99)
parser.add_argument('--positiveCaseWeight', help='For the loss BCEWithLogitsLoss, the weight of positive cases. Default: 3.0', type=float, default=3.0)
parser.add_argument('--numberOfLatentVariables', help='The number of latent variables for the autoencoder. Default: 20', type=int, default=21)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

print ("args.numberOfLatentVariables = {}".format(args.numberOfLatentVariables))
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0

def GenerateRandomPositions(
        numberOfPositions,
        playerList,
        authority,
        minimumNumberOfMoves,
        maximumNumberOfMoves
        ):
    positionsList = []
    while len(positionsList) < numberOfPositions:
        #maximumNumberOfMovesForThisSimulation = numpy.random.randint(minimumNumberOfMoves,
        #    maximumNumberOfMoves)
        gamePositionsList, winner = expectedMoveValues.SimulateAGame(
            playerList,
            authority=authority,
            neuralNetwork=None,
            softMaxTemperatureForSelfPlayEvaluation=1.0,  # Will be ignored
            epsilon=0.1,
            maximumNumberOfMoves=maximumNumberOfMoves,#ForThisSimulation,
            startingPosition=None,
            opponentPlaysRandomly=True
        )
        if len(gamePositionsList) >= minimumNumberOfMoves:
            selectedNdx = numpy.random.randint(len(gamePositionsList))
            positionsList.append(gamePositionsList[selectedNdx])  # Keep the selected position
    return positionsList

def main():
    print("learnTicTacToeAutoencoder.py main()")

    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    playersList = authority.PlayersList()

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
            bodyStructure=[(3, 32, 1)],#, (3, 64, 2)],#, (5, 16), (5, 16)],
            numberOfLatentVariables=args.numberOfLatentVariables,
            zeroPadding=False
        )

    # Create the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, neuralNetwork.parameters()),
                                 lr=args.learningRate, betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.positiveCaseWeight]))  # torch.nn.MSELoss()

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

        logging.info("Generating {} training random positions".format(args.numberOfPositionsForTraining))
        trainingPositionsList = GenerateRandomPositions(args.numberOfPositionsForTraining, playersList, authority,
                                                        minimumNumberOfMovesForInitialPositions,
                                                        args.maximumNumberOfMovesForPositions)
        #print ("trainingPositionsList =\n{}".format(trainingPositionsList))
        logging.info("Generating {} validation random positions".format(args.numberOfPositionsForValidation))
        validationPositionsList = GenerateRandomPositions(args.numberOfPositionsForValidation, playersList, authority,
                                                          minimumNumberOfMovesForInitialPositions,
                                                          args.maximumNumberOfMovesForPositions)

        trainingLossSum = 0.0
        minibatchIndicesList = utilities.MinibatchIndices(len(trainingPositionsList), args.minibatchSize)

        logging.info("Going through the minibatch")
        for minibatchNdx in range(len(minibatchIndicesList)):
            print('.', end='', flush=True)

            minibatchPositions = []
            for index in minibatchIndicesList[minibatchNdx]:
                # logging.debug("len(positionStatisticsList[{}]) = {}".format(index, len(positionStatisticsList[index])))
                minibatchPositions.append(trainingPositionsList[index])
            minibatchPositionsTensor = utilities.MinibatchTensor(minibatchPositions)
            minibatchTargetPositionsTensor = utilities.MinibatchTensor(
                minibatchPositions)  # Autoencoder => target output = input

            optimizer.zero_grad()

            # Forward pass
            outputTensor = neuralNetwork(minibatchPositionsTensor)

            # Calculate the error and backpropagate
            trainingLoss = loss(outputTensor, minibatchTargetPositionsTensor)
            # logging.debug("trainingLoss.item() = {}".format(trainingLoss.item()))

            trainingLoss.backward()
            trainingLossSum += trainingLoss.item()

            # Move in the gradient descent direction
            optimizer.step()

        averageTrainingLoss = trainingLossSum / len(minibatchIndicesList)

        # Compute the validation loss
        neuralNetwork.eval()
        validationPositionsTensor = utilities.MinibatchTensor(validationPositionsList)
        validationOutputTensor = neuralNetwork(validationPositionsTensor)
        validationLoss = loss(validationOutputTensor,
                              validationPositionsTensor).item()  # Autoencoder => target output = input

        # Compare the output tensor converted to one-hot with the target
        oneHotValidationOutputTensor = position.ConvertToOneHotPositionTensor(validationOutputTensor)
        numberOfErrors = torch.nonzero(validationPositionsTensor.long() - oneHotValidationOutputTensor).shape[0]
        errorRate = numberOfErrors / max(torch.nonzero(validationPositionsTensor).shape[0], 0)

        print(" * ")
        logging.info(
            "Epoch {}: averageTrainingLoss = {}\tvalidationLoss = {}\terrorRate = {}".format(epoch, averageTrainingLoss,
                                                                                             validationLoss, errorRate))
        epochLossFile.write(
            str(epoch) + ',' + str(averageTrainingLoss) + ',' + str(validationLoss) + ',' + str(errorRate) + '\n')

        # Save the neural network
        neuralNetwork.Save(args.outputDirectory, 'tictactoeAutoencoder_' + str(epoch))

        # Update the learning rate
        learningRate = learningRate * args.learningRateExponentialDecay
        utilities.adjust_lr(optimizer, learningRate)

if __name__ == '__main__':
    main()