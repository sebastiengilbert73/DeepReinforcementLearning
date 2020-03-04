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
import PCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--recomputingPeriod', help='The period of recomputation of the training examples. Default: 30', type=int, default=30)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--outputFilenamesPrefix', help='The prefix of the outputs filenames. Default: netEnsembleMbr_', default='netEnsembleMbr_')
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--startWithAutoencoder', help='The autoencoder whose encoder will be used. Default: None', default=None)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 7', type=int, default=7)
parser.add_argument('--numberOfPositionsForTraining', help='The number of positions for training per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfPositionsForValidation', help='The number of positions for validation per epoch. Default: 128', type=int, default=128)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.2', type=float, default=0.2)
parser.add_argument('--numberOfSimulations', help='For each starting position, the number of simulations to evaluate the position value. Default: 30', type=int, default=30)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--dropoutRatio', help='The dropout ratio. Default: 0.25', type=float, default=0.25)
parser.add_argument('--numberOfGamesAgainstARandomPlayer', help='The number of games, when playing against a random player. Default: 30', type=int, default=30)
parser.add_argument('--numberOfNeuralNetworks', help='Maximum number of neural networks to generate. Default: 31', type=int, default=31)
parser.add_argument('--lossRateThreshold', help='The inverse threshold of loss rate against a random player, to keep the neural network. Default: 0.2', type=float, default=0.2)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0

def StartingPositionsInPairsOfPossibleOptions(startingPositionsList, authority):
    playersList = authority.PlayersList()
    positionShape = startingPositionsList[0].shape
    startingPositionsTensor = torch.zeros(2 * len(startingPositionsList), 2 * positionShape[0],
                                          positionShape[1], positionShape[2], positionShape[3])
    augmentedStartingPositionsList = []

    for pairNdx in range(len(startingPositionsList) ):
        rootPosition = startingPositionsList[pairNdx]
        candidatePositionWinnerPairsList = utilities.LegalCandidatePositionsAfterMove(authority, rootPosition, playersList[0])
        position0 = candidatePositionWinnerPairsList[0][0]
        position1 = candidatePositionWinnerPairsList[0][0]
        if len (candidatePositionWinnerPairsList) > 1:
            indices = numpy.arange(len(candidatePositionWinnerPairsList))
            numpy.random.shuffle(indices)
            #if candidatePositionWinnerPairsList[indices[0]][1] is None and \
            #        candidatePositionWinnerPairsList[indices[1]][1] is None:
            position0 = candidatePositionWinnerPairsList[indices[0]][0]
            position1 = candidatePositionWinnerPairsList[indices[1]][0]
        if torch.equal(position0, position1):
            print("StartingPositionsInPairsOfPossibleOptions(): position0 and position1 are identical")
            print ("position0 = \n{}".format(position0))
            print ("rootPosition = \n{}".format(rootPosition))
        positionPair01Tsr = torch.zeros(2 * positionShape[0], positionShape[1], positionShape[2], positionShape[3])
        positionPair01Tsr[0: positionShape[0]] = position0
        positionPair01Tsr[positionShape[0]:] = position1

        positionPair10Tsr = torch.zeros(2 * positionShape[0], positionShape[1], positionShape[2], positionShape[3])
        positionPair10Tsr[0: positionShape[0]] = position1
        positionPair10Tsr[positionShape[0]:] = position0

        startingPositionsTensor[2 * pairNdx] = positionPair01Tsr
        startingPositionsTensor[2 * pairNdx + 1] = positionPair10Tsr

        augmentedStartingPositionsList.append(position0)
        augmentedStartingPositionsList.append(position1)
    return startingPositionsTensor, augmentedStartingPositionsList

def PairWinnerIndexTensor(pairWinnerIndexList):
    pairWinnerIndexTsr = torch.zeros(2 * len(pairWinnerIndexList) ).long()
    for pairNdx in range(len(pairWinnerIndexList)):
        winnerNdx = pairWinnerIndexList[pairNdx]
        if winnerNdx == 0:
            pairWinnerIndexTsr[2 * pairNdx] = 0
            pairWinnerIndexTsr[2 * pairNdx + 1] = 1
        else:
            pairWinnerIndexTsr[2 * pairNdx] = 1
            pairWinnerIndexTsr[2 * pairNdx + 1] = 0
    return pairWinnerIndexTsr

def Accuracy(predictionTsr, targetTsr):
    if predictionTsr.shape[0] != targetTsr.shape[0]:
        raise ValueError("The number of predictions ({}) doesn't match the number of targets ({})".format(predictionTsr.shape[0], targetTsr.shape[0]))
    if predictionTsr.shape[0] > 0:
        numberOfCorrectPredictions = 0
        predictionIndexTsr = torch.argmax(predictionTsr, dim=1)
        for sampleNdx in range(predictionTsr.shape[0]):
            if predictionIndexTsr[sampleNdx] == targetTsr[sampleNdx]:
                numberOfCorrectPredictions += 1
        return float(numberOfCorrectPredictions) / predictionTsr.shape[0]
    else:
        return 0.0

def main():
    print ("generateComparisonNetEnsemble.py: main()")

    authority = tictactoe.Authority()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        raise NotImplementedError("main(): Start with a neural network is not implemented...")
    else:
        if args.startWithAutoencoder is not None:
            autoencoderNet = autoencoder.position.Net()
            autoencoderNet.Load(args.startWithAutoencoder)

        else:
            raise NotImplementedError("main(): Starting without an autoencoder is not implemented...")



    # Loss function
    loss = torch.nn.CrossEntropyLoss()  # The neural network is a binary classifier

    # Initial learning rate
    learningRate = args.learningRate





    epsilon = args.epsilon

    for neuralNetworkNdx in range(1, args.numberOfNeuralNetworks + 1):
        logging.info(" ------ Starting training of neural network # {} ------".format(neuralNetworkNdx))

        # Output monitoring file
        epochLossFile = open(os.path.join(args.outputDirectory, 'netEnsembleEpochLoss.csv'), "w",
                             buffering=1)  # Flush the buffer at each line
        epochLossFile.write(
            "epoch,trainingLoss,validationLoss,validationAccuracy,averageReward,winRate,drawRate,lossRate\n")

        decoderClassifier = ComparisonNet.BuildADecoderClassifierFromAnAutoencoder(
            autoencoderNet, dropoutRatio=args.dropoutRatio)

        # Create the optimizer
        for name, param in decoderClassifier.named_parameters():
            if 'decoding' in name:
                param.requires_grad = True
            else:
                param.requires_grad = True
            print("name = {}; param.requires_grad = {}".format(name, param.requires_grad))

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, decoderClassifier.parameters()),
                                     lr=learningRate,
                                     betas=(0.5, 0.999))

        terminalConditionIsReached = False
        epoch = 1
        while epoch <= args.numberOfEpochs and not terminalConditionIsReached:
            logging.info("Epoch {}".format(epoch))

            if epoch > 1 and epoch % 50 == 1:
                epsilon = epsilon / 2

            # Generate positions
            if epoch % args.recomputingPeriod == 1:
                minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
                maximumNumberOfMovesForInitialPositions = args.maximumNumberOfMovesForInitialPositions
                logging.info("Generating positions...")
                startingPositionsList = Comparison.SimulateRandomGames(authority,
                                                                       minimumNumberOfMovesForInitialPositions,
                                                                       maximumNumberOfMovesForInitialPositions,
                                                                       args.numberOfPositionsForTraining,
                                                                       swapIfOddNumberOfMoves=True)

                startingPositionsTensor, augmentedStartingPositionsList = StartingPositionsInPairsOfPossibleOptions(
                    startingPositionsList, authority)
                logging.info("Comparing starting position pairs...")
                decoderClassifier.eval()
                pairWinnerIndexList = Comparison.ComparePositionPairs(authority, decoderClassifier,
                                                                      augmentedStartingPositionsList,
                                                                      args.numberOfSimulations,
                                                                      epsilon=0,
                                                                      playerToEpsilonDict={playerList[0]: epsilon,
                                                                                           playerList[1]: epsilon})
                # print ("pairWinnerIndexList = {}".format(pairWinnerIndexList))

                pairWinnerIndexTsr = PairWinnerIndexTensor(pairWinnerIndexList)

            decoderClassifier.train()

            # Since the samples are generated dynamically, there is no need for minibatches: all samples are always new
            optimizer.zero_grad()

            # Forward pass
            outputTensor = decoderClassifier(startingPositionsTensor)

            # Calculate the error and backpropagate
            trainingLoss = loss(outputTensor, pairWinnerIndexTsr)
            logging.info("trainingLoss.item() = {}".format(trainingLoss.item()))

            trainingLoss.backward()

            # Move in the gradient descent direction
            optimizer.step()

            # ******************  Validation ******************
            decoderClassifier.eval()

            if epoch % 50 == 1:
                logging.info("Generating validation positions...")
                validationStartingPositionsList = Comparison.SimulateRandomGames(authority,
                                                                                 minimumNumberOfMovesForInitialPositions,
                                                                                 maximumNumberOfMovesForInitialPositions,
                                                                                 args.numberOfPositionsForValidation,
                                                                                 swapIfOddNumberOfMoves=True)

                validationStartingPositionsTensor, validationAugmentedStartingPositionsList = \
                    StartingPositionsInPairsOfPossibleOptions(validationStartingPositionsList, authority)

                logging.info("Comparing validation starting position pairs...")
                validationPairWinnerIndexList = Comparison.ComparePositionPairs(authority, decoderClassifier,
                                                                                validationAugmentedStartingPositionsList,
                                                                                args.numberOfSimulations,
                                                                                epsilon=0,
                                                                                playerToEpsilonDict={
                                                                                    playerList[0]: epsilon, playerList[
                                                                                        1]: epsilon})  # Start with purely random games (epsilon = 1)
                validationPairWinnerIndexTsr = PairWinnerIndexTensor(validationPairWinnerIndexList)

            # Forward pass
            validationOutputTensor = decoderClassifier(validationStartingPositionsTensor)

            # Calculate the validation error
            validationLoss = loss(validationOutputTensor, validationPairWinnerIndexTsr)

            # Calculate the validation accuracy
            validationAccuracy = Accuracy(validationOutputTensor, validationPairWinnerIndexTsr)

            logging.info("validationLoss.item() = {};    validationAccuracy = {}".format(validationLoss.item(),
                                                                                         validationAccuracy))

            # Play against a random player
            logging.info("Play against a random player...")
            (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer,
             numberOfDraws) = Comparison.SimulateGamesAgainstARandomPlayer(
                decoderClassifier, authority, args.numberOfGamesAgainstARandomPlayer)

            winRate = numberOfWinsForEvaluator / (
                        numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
            lossRate = numberOfWinsForRandomPlayer / (
                    numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
            drawRate = numberOfDraws / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
            logging.info(
                "Against a random player, winRate = {}; drawRate = {}; lossRate = {}".format(winRate, drawRate,
                                                                                             lossRate))

            epochLossFile.write(
                str(epoch) + ',' + str(trainingLoss.item()) + ',' + str(validationLoss.item()) + ',' + str(
                    validationAccuracy) + ',' +
                str(winRate - lossRate) + ',' + str(winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

            if lossRate <= args.lossRateThreshold:
                # Check with more random games
                (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer, numberOfDraws) = Comparison.SimulateGamesAgainstARandomPlayer(decoderClassifier, authority, 100)

                winRate = numberOfWinsForEvaluator / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
                lossRate = numberOfWinsForRandomPlayer / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
                drawRate = numberOfDraws / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
                logging.info("***** With 100 games against a random player, winRate = {}; drawRate = {}; lossRate = {} *****".format(winRate, drawRate,
                                                                                         lossRate))

            if lossRate <= args.lossRateThreshold:
                terminalConditionIsReached = True
                filepath = os.path.join(args.outputDirectory, args.outputFilenamesPrefix + str(neuralNetworkNdx) + '.bin')
                decoderClassifier.Save(filepath)

            epoch += 1

if __name__ == '__main__':
    main()