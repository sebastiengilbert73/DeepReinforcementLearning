import argparse
import numpy
import torch
import os
import ast
import tictactoe
import logging
import autoencoder.position # autoencoder
import Comparison
import ComparisonNet
import numpy.random
import utilities
import pandas
import winRatesRegression


parser = argparse.ArgumentParser()
parser.add_argument('encodingWinRatesFilepath', help='The filepath to the file containing the embedding and the win rates')
parser.add_argument('autoencoderFilepath', help='The filepath to the autoencoder')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--minibatchSize', help='The size of the minibatch. Default: 16', type=int, default=16)
parser.add_argument('--neuralNetworkLayerSizesList', help="The list of layer sizes. Default: '[8, 4, 3]'", default='[8, 4, 3]')
parser.add_argument('--dropoutRatio', help='The dropout ratio. Default: 0.1', type=float, default=0.1)
parser.add_argument('--numberOfGamesAgainstRandomPlayer', help='The number of games played against a random player. Default: 100', type=int, default=100)
parser.add_argument('--numberOfRuns', help='The number of times the epochs are run. Default: 100', type=int, default=100)
parser.add_argument('--stageIndex', help='The stage index for the saving filename of regressors. Default: 0', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
neuralNetworkLayerSizesList = ast.literal_eval(args.neuralNetworkLayerSizesList)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def DataTensors(embeddingWinRatesDF):
    #numberOfSamples = embeddingWinRatesDF.shape[0]
    numberOfColumns = embeddingWinRatesDF.shape[1]

    attributesDF = embeddingWinRatesDF.iloc[:, 0: numberOfColumns - 3]
    attributesTsr = torch.tensor(attributesDF.values)
    targetWinRatesDF = embeddingWinRatesDF.iloc[:, numberOfColumns - 3: numberOfColumns]
    targetWinRatesTsr = torch.tensor(targetWinRatesDF.values)
    return attributesTsr, targetWinRatesTsr

def SplitTrainingAndValidation(attributesTsr, targetsTsr, validationProportion):
    if attributesTsr.shape[0] != targetsTsr.shape[0]:
        raise ValueError("SplitTrainingAndValidation(): attributesTsr.shape[0] ({}) != targetsTsr.shape[0] ({})".format(attributesTsr.shape[0], targetsTsr.shape[0]))
    numberOfSamples = attributesTsr.shape[0]
    indicesList = numpy.arange(numberOfSamples)
    numpy.random.shuffle(indicesList)
    numberOfValidationSamples = int(validationProportion * numberOfSamples)
    validationAttributesTsr = attributesTsr[0: numberOfValidationSamples]
    validationTargetsTsr = targetsTsr[0: numberOfValidationSamples]
    trainingAttributesTsr = attributesTsr[numberOfValidationSamples:]
    trainingTargetsTsr = targetsTsr[numberOfValidationSamples: ]
    return (trainingAttributesTsr.float(), trainingTargetsTsr.float(), validationAttributesTsr.float(), validationTargetsTsr.float())

def MinibatchTensors(attributesTsr, targetsTsr, indicesList):
    if attributesTsr.shape[0] != targetsTsr.shape[0]:
        raise ValueError("MinibatchTensors(): attributesTsr.shape[0] ({}) != targetsTsr.shape[0] ({})".format(attributesTsr.shape[0], targetsTsr.shape[0]))
    minibatchAttributesTsr = torch.zeros(len(indicesList), attributesTsr.shape[1])
    minibatchTargetsTsr = torch.zeros(len(indicesList), targetsTsr.shape[1])
    for rowNdx in range(len(indicesList)):
        chosenNdx = indicesList[rowNdx]
        minibatchAttributesTsr[rowNdx] = attributesTsr[chosenNdx]
        minibatchTargetsTsr[rowNdx] = targetsTsr[chosenNdx]
    return minibatchAttributesTsr, minibatchTargetsTsr

def main():
    logging.info("learnWinRatesFromEncoding.py main()")

    # Load the data file
    embeddingWinRatesDF = pandas.read_csv(args.encodingWinRatesFilepath)
    #print ("embeddingWinRatesDF = {}".format(embeddingWinRatesDF))

    attributesTsr, targetWinRatesTsr = DataTensors(embeddingWinRatesDF)
    numberOfAttributes = attributesTsr.shape[1]
    numberOfSamples = attributesTsr.shape[0]

    # Split training and validation samples
    (trainingAttributesTsr, trainingTargetsTsr, validationAttributesTsr, validationTargetsTsr) = SplitTrainingAndValidation(
        attributesTsr, targetWinRatesTsr, validationProportion=0.2
    )

    #print ("attributesTsr = {}".format(attributesTsr))
    #print ("targetWinRatesTsr = {}".format(targetWinRatesTsr))

    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    # Loss function
    # loss = torch.nn.MSELoss()  # The neural network is a regressor
    loss = torch.nn.SmoothL1Loss()

    # Initial learning rate
    learningRate = args.learningRate


    for runNdx in range(args.numberOfRuns):
        logging.info(" +++++++++++++++ Run {} +++++++++++++++".format(runNdx))

        # Create the neural network
        regressor = winRatesRegression.Net(
            inputNumberOfAttributes=numberOfAttributes,
            bodyStructureList=neuralNetworkLayerSizesList,
            dropoutRatio=args.dropoutRatio
        )

        # Create the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, regressor.parameters()),
                                     lr=args.learningRate,
                                     betas=(0.5, 0.999))



        # Output monitoring file
        epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                             buffering=1)  # Flush the buffer at each line
        epochLossFile.write(
            "epoch,trainingLoss,validationLoss\n")

        # Autoencoder
        autoencoderNet = autoencoder.position.Net()
        autoencoderNet.Load(args.autoencoderFilepath)

        for epoch in range(1, args.numberOfEpochs + 1):
            logging.info ("Epoch {}".format(epoch))
            regressor.train()

            indicesList = numpy.arange(trainingAttributesTsr.shape[0])
            numpy.random.shuffle(indicesList)

            numberOfminibatches = len(indicesList) // args.minibatchSize
            trainigLossSum = 0
            for minibatchNdx in range(numberOfminibatches):
                print ('.', end='', flush=True)

                indexNdx0 = minibatchNdx * args.minibatchSize
                indexNdx1 = (minibatchNdx + 1) * args.minibatchSize
                minibatchIndicesList = indicesList[indexNdx0 : indexNdx1]
                #print ("minibatchIndicesList = {}".format(minibatchIndicesList))
                minibatchAttributesTsr, minibatchTargetsTsr = MinibatchTensors(trainingAttributesTsr, trainingTargetsTsr, minibatchIndicesList)

                optimizer.zero_grad()

                # Forward pass
                minibatchOutputTensor = regressor(minibatchAttributesTsr)

                # Calculate the error and backpropagate
                trainingLoss = loss(minibatchOutputTensor, minibatchTargetsTsr)
                #logging.info("trainingLoss.item() = {}".format(trainingLoss.item()))
                trainigLossSum += trainingLoss.item()

                trainingLoss.backward()

                # Move in the gradient descent direction
                optimizer.step()

            averageTrainigLoss = trainigLossSum / numberOfminibatches

            # ******************  Validation ******************
            regressor.eval()
            validationOutputTsr = regressor(validationAttributesTsr)
            validationLoss = loss(validationOutputTsr, validationTargetsTsr).item()

            logging.info("averageTrainigLoss = {}; validationLoss = {}".format(averageTrainigLoss, validationLoss))
            epochLossFile.write("{},{},{}\n".format(epoch, averageTrainigLoss, validationLoss))

            # ****************** Compare with a random player **************************
            if epoch % 10 == 1 or epoch == args.numberOfEpochs:
                (numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws) = winRatesRegression.SimulateGamesAgainstARandomPlayer(
                    regressor, autoencoderNet, authority, args.numberOfGamesAgainstRandomPlayer, None)
                logging.info ("numberOfWinsForRegressor = {}; numberOfWinsForRandomPlayer = {}; numberOfDraws = {}".format(
                    numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws))
        # Save the neural network
        regressor.Save(os.path.join(args.outputDirectory, 'regressor_' + str(args.stageIndex) + '_' + str(runNdx) + '.bin'))


if __name__ == '__main__':
    main()