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


parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--startWithNeuralNetwork', help='The starting neural network weights. Default: None', default=None)
parser.add_argument('--startWithAutoencoder', help='The autoencoder whose encoder will be used. Default: None', default=None)
parser.add_argument('--maximumNumberOfMovesForInitialPositions', help='The maximum number of moves in the initial positions. Default: 7', type=int, default=7)
parser.add_argument('--numberOfPositionsForTraining', help='The number of positions for training per epoch. Default: 128', type=int, default=128)
parser.add_argument('--numberOfPositionsForValidation', help='The number of positions for validation per epoch. Default: 128', type=int, default=128)
parser.add_argument('--epsilon', help='Probability to do a random move while generating move statistics. Default: 0.2', type=float, default=0.2)
parser.add_argument('--numberOfSimulations', help='For each starting position, the number of simulations to evaluate the position value. Default: 30', type=int, default=30)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
parser.add_argument('--decodingLayerSizesList', help="The list of decoding layer sizes. Default: '[8, 2]'", default='[8, 2]')
parser.add_argument('--numberOfGamesAgainstARandomPlayer', help='The number of games, when playing against a random player. Default: 30', type=int, default=30)


args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
decodingLayerSizesList = ast.literal_eval(args.decodingLayerSizesList)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def MinimumNumberOfMovesForInitialPositions(epoch):
    return 0

def StartingPositionsTensor(startingPositionsList):
    if len(startingPositionsList) % 2 == 1:
        startingPositionsList = startingPositionsList[: -1] # Make sure we have an even number of positions
    positionShape = startingPositionsList[0].shape
    startingPositionsTensor = torch.zeros(len(startingPositionsList), 2 * positionShape[0], positionShape[1],
                                       positionShape[2], positionShape[3]) # We 'll swap the pairs, in order to have a symmetric behavior: p0 > p1 -> p1 < p0
    for pairNdx in range(len(startingPositionsList)//2):
        positionPair01Tsr = torch.zeros(2 * positionShape[0], positionShape[1], positionShape[2], positionShape[3])
        positionPair01Tsr[0: positionShape[0] ] = startingPositionsList[2 * pairNdx]
        positionPair01Tsr[positionShape[0]:] = startingPositionsList[2 * pairNdx + 1]

        positionPair10Tsr = torch.zeros(2 * positionShape[0], positionShape[1], positionShape[2], positionShape[3])
        positionPair10Tsr[0: positionShape[0]] = startingPositionsList[2 * pairNdx + 1]
        positionPair10Tsr[positionShape[0]:] = startingPositionsList[2 * pairNdx]

        startingPositionsTensor[2 * pairNdx] = positionPair01Tsr
        startingPositionsTensor[2 * pairNdx + 1] = positionPair10Tsr
    return startingPositionsTensor

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
            if candidatePositionWinnerPairsList[indices[0]][1] is None and \
                    candidatePositionWinnerPairsList[indices[1]][1] is None:
                position0 = candidatePositionWinnerPairsList[indices[0]][0]
                position1 = candidatePositionWinnerPairsList[indices[1]][0]
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

def Majority(startingPositionsList):
    numberOfMajorityX = 0
    numberOfMajorityO = 0
    numberOfEqualities = 0

    for startingPosition in startingPositionsList:
        numberOfX = torch.nonzero(startingPosition[0]).shape[0]
        numberOfO = torch.nonzero(startingPosition[1]).shape[0]
        if numberOfX > numberOfO:
            numberOfMajorityX += 1
        elif numberOfO > numberOfX:
            numberOfMajorityO += 1
        else:
            numberOfEqualities += 1
    return numberOfMajorityX, numberOfMajorityO, numberOfEqualities


def main():
    print ("learnTicTacToeWithComparisonNet.py main()")

    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playerList = authority.PlayersList()

    if args.startWithNeuralNetwork is not None:
        raise NotImplementedError("main(): Start with a neural network is not implemented...")
    else:
        if args.startWithAutoencoder is not None:
            autoencoderNet = autoencoder.position.Net()
            autoencoderNet.Load(args.startWithAutoencoder)

            decoderClassifier = ComparisonNet.BuildADecoderClassifierFromAnAutoencoder(
                autoencoderNet)
        else:
            raise NotImplementedError("main(): Starting without an autoencoder is not implemented...")

    # Create the optimizer
    logging.debug(decoderClassifier)
    for name, param in decoderClassifier.named_parameters():
        if 'decoding' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print("name = {}; param.requires_grad = {}".format(name, param.requires_grad))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, decoderClassifier.parameters()), lr=args.learningRate,
                                 betas=(0.5, 0.999))

    # Loss function
    loss = torch.nn.CrossEntropyLoss() # The neural network is a binary classifier

    # Initial learning rate
    learningRate = args.learningRate

    # Output monitoring file
    epochLossFile = open(os.path.join(args.outputDirectory, 'epochLoss.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    epochLossFile.write(
        "epoch,trainingLoss,validationLoss,averageReward,winRate,drawRate,lossRate\n")

    # First game with a random player, before any training
    (numberOfWinsForComparator, numberOfWinsForRandomPlayer,
     numberOfDraws) = Comparison.SimulateGamesAgainstARandomPlayer(
        decoderClassifier, authority, args.numberOfGamesAgainstARandomPlayer)
    print ("(numberOfWinsForComparator, numberOfWinsForRandomPlayer, numberOfDraws) = ({}, {}, {})".format(numberOfWinsForComparator, numberOfWinsForRandomPlayer,
     numberOfDraws))

    winRate = numberOfWinsForComparator / (numberOfWinsForComparator + numberOfWinsForRandomPlayer + numberOfDraws)
    lossRate = numberOfWinsForRandomPlayer / (numberOfWinsForComparator + numberOfWinsForRandomPlayer + numberOfDraws)
    drawRate = numberOfDraws / (numberOfWinsForComparator + numberOfWinsForRandomPlayer + numberOfDraws)
    logging.info(
        "Against a random player, winRate = {}; drawRate = {}; lossRate = {}".format(winRate, drawRate, lossRate))

    epochLossFile.write(
        '0' + ',' + '-' + ',' + '-' + ',' + str(winRate - lossRate) + ',' + str(
            winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')

    playerToEpsilonDict = {playerList[0]: args.epsilon, playerList[1]: args.epsilon}

    for epoch in range(1, args.numberOfEpochs + 1):
        logging.info ("Epoch {}".format(epoch))
        decoderClassifier.train()

        # Generate positions
        minimumNumberOfMovesForInitialPositions = MinimumNumberOfMovesForInitialPositions(epoch)
        maximumNumberOfMovesForInitialPositions = args.maximumNumberOfMovesForInitialPositions
        logging.info("Generating positions...")
        startingPositionsList = Comparison.SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions,
                                                    maximumNumberOfMovesForInitialPositions,
                                                    args.numberOfPositionsForTraining,
                                                    swapIfOddNumberOfMoves=True)

        """for startingPositionNdx in range(len(startingPositionsList)):
            if numpy.random.random() >= 0.5:
                #print ("main(): swapped...")
                swappedPosition = authority.SwapPositions(startingPositionsList[startingPositionNdx], playerList[0], playerList[1])
                startingPositionsList[startingPositionNdx] = swappedPosition
        """

        numberOfMajorityX, numberOfMajorityO, numberOfEqualities = Majority(startingPositionsList)
        print ("numberOfMajorityX = {}; numberOfMajorityO = {}; numberOfEqualities = {}".format(numberOfMajorityX, numberOfMajorityO, numberOfEqualities))
        #print ("main(): startingPositionsList = {}".format(startingPositionsList))

        startingPositionsTensor, augmentedStartingPositionsList = StartingPositionsInPairsOfPossibleOptions(startingPositionsList, authority)
        #print ("main(): augmentedStartingPositionsList = {}".format(augmentedStartingPositionsList))
        #print ("main(): startingPositionsTensor.shape = {}".format(startingPositionsTensor.shape))
        #print ("main(): startingPositionsTensor = {}".format(startingPositionsTensor))

        logging.info("Comparing starting position pairs...")
        pairWinnerIndexList = Comparison.ComparePositionPairs(authority, decoderClassifier,
                                                              augmentedStartingPositionsList,
                                                              args.numberOfSimulations,
                                                              args.epsilon)
        #print ("pairWinnerIndexList = {}".format(pairWinnerIndexList))

        pairWinnerIndexTsr = PairWinnerIndexTensor(pairWinnerIndexList)
        #print ("main(): pairWinnerIndexTsr.shape = {}".format(pairWinnerIndexTsr.shape))
        #print ("main(): pairWinnerIndexTsr = {}".format(pairWinnerIndexTsr))

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


        # Validation
        decoderClassifier.eval()

        logging.info("Generating validation positions...")
        validationStartingPositionsList = Comparison.SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions,
                                                               maximumNumberOfMovesForInitialPositions,
                                                               args.numberOfPositionsForValidation,
                                                               swapIfOddNumberOfMoves=True)
        """for validationStartingPositionNdx in range(len(validationStartingPositionsList)):
            if numpy.random.random() >= 0.5:
                swappedPosition = authority.SwapPositions(validationStartingPositionsList[validationStartingPositionNdx], playerList[0], playerList[1])
                validationStartingPositionsList[validationStartingPositionNdx] = swappedPosition
        """
        # print ("main(): startingPositionsList = {}".format(startingPositionsList))

        validationStartingPositionsTensor, validationAugmentedStartingPositionsList = \
            StartingPositionsInPairsOfPossibleOptions(validationStartingPositionsList, authority)

        logging.info("Comparing validation starting position pairs...")
        validationPairWinnerIndexList = Comparison.ComparePositionPairs(authority, decoderClassifier,
                                                              validationAugmentedStartingPositionsList,
                                                              args.numberOfSimulations,
                                                              args.epsilon)
        validationPairWinnerIndexTsr = PairWinnerIndexTensor(validationPairWinnerIndexList)

        # Forward pass
        validationOutputTensor = decoderClassifier(validationStartingPositionsTensor)

        # Calculate the validation error
        validationLoss = loss(validationOutputTensor, validationPairWinnerIndexTsr)
        logging.info("validationLoss.item() = {}".format(validationLoss.item()))

        logging.info("Play against a random player...")
        # Play against a random player
        (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer,
         numberOfDraws) = Comparison.SimulateGamesAgainstARandomPlayer(
        decoderClassifier, authority, args.numberOfGamesAgainstARandomPlayer)

        winRate = numberOfWinsForEvaluator / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
        lossRate = numberOfWinsForRandomPlayer / (
                numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
        drawRate = numberOfDraws / (numberOfWinsForEvaluator + numberOfWinsForRandomPlayer + numberOfDraws)
        logging.info(
            "Against a random player, winRate = {}; drawRate = {}; lossRate = {}".format(winRate, drawRate, lossRate))

        epochLossFile.write(
            str(epoch) + ',' + str(trainingLoss.item()) + ',' + str(validationLoss.item()) + ',' + str(
                winRate - lossRate) + ',' + str(
                winRate) + ',' + str(drawRate) + ',' + str(lossRate) + '\n')
        if epoch % 10 == 0:
            filepath = os.path.join(args.outputDirectory, 'tictactoe_' + str(epoch) + '.bin')
            decoderClassifier.Save(filepath)

            epsilon0GamePositionsList, epsilon0GameWinner = Comparison.SimulateAGame(decoderClassifier, authority)
            for position in epsilon0GamePositionsList:
                authority.Display(position)
                print(".............\n")

        if epoch % 100 == 0:
            learningRate = learningRate/3
            for param_group in optimizer.param_groups:
                param_group['lr'] = learningRate

if __name__ == '__main__':
    main()