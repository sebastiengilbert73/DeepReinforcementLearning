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
import io


parser = argparse.ArgumentParser()
parser.add_argument('neuralNetworksFilepathPrefix', help='The prefix to the filepath of the member neural networks')
parser.add_argument('--testToDo', help="The test you want to do. Default: 'runGames'", default='runGames')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

#neuralNetworkIndicesList = [1, 8, 12, 14, 18, 20, 23, 26, 28]

def main():
    print ("testNetEnsemble.py main()")
    # Get the neural networks filepaths
    filesDirectory = os.path.dirname(args.neuralNetworksFilepathPrefix)
    filesList = [os.path.join(filesDirectory, f) for f in os.listdir(filesDirectory) if
                 os.path.isfile(os.path.join(filesDirectory, f))]
    filesList = [f for f in filesList if args.neuralNetworksFilepathPrefix in f]
    #print ("filesList = {}".format(filesList))
    #print ("len(filesList) = {}".format(len(filesList)))

    neuralNetworksList = [ Comparison.Load(filepath) for filepath in filesList]
    #netEnsemble = Comparison.ComparatorsEnsemble(neuralNetworksList)

    authority = tictactoe.Authority()
    positionTsrShape = authority.PositionTensorShape()
    playersList = authority.PlayersList()

    if args.testToDo == 'runGames':
        for numberOfNeuralNetworks in range(1, len(neuralNetworksList) + 1):
            limitedNetList = []
            for neuralNetNdx in range(numberOfNeuralNetworks):
                limitedNetList.append(neuralNetworksList[neuralNetNdx])
            netEnsemble = Comparison.ComparatorsEnsemble(limitedNetList)

            if True: #numberOfNeuralNetworks % 10 == 1:
                (numberOfWinsForComparator, numberOfWinsForRandomPlayer, numberOfDraws) = Comparison.SimulateGamesAgainstARandomPlayer(
                    netEnsemble, authority, 100
                )

                logging.info("numberOfNeuralNetworks = {}; numberOfWinsForComparator = {}; numberOfWinsForRandomPlayer = {}; numberOfDraws = {}".format(
                    numberOfNeuralNetworks, numberOfWinsForComparator, numberOfWinsForRandomPlayer, numberOfDraws
                ))
    elif args.testToDo == 'evaluatePosition':
        positionToEvaluate = torch.zeros(positionTsrShape)
        positionToEvaluate[0, 0, 0, 2] = 1
        positionToEvaluate[1, 0, 0, 0] = 1
        authority.Display(positionToEvaluate)

        numberOfSimulations = 30
        numberOfEpsilonSteps = 11
        netEnsemble = Comparison.ComparatorsEnsemble(neuralNetworksList)

        for epsilonNdx in range(numberOfEpsilonSteps):
            epsilon = epsilonNdx * 1.0/(numberOfEpsilonSteps - 1)
            logging.info("epsilon = {}".format(epsilon))

            numberOfWinsForPlayer0 = 0
            numberOfWinsForPlayer1 = 0
            numberOfDraws = 0
            for simulationNdx in range(numberOfSimulations):
                (positionsList, winner) = Comparison.SimulateAGame(netEnsemble, authority,
                                                                   startingPosition=positionToEvaluate,
                                                                   nextPlayer=playersList[0],
                                                                   playerToEpsilonDict={playersList[0]: epsilon, playersList[1]: epsilon})
                if winner == playersList[0]:
                    numberOfWinsForPlayer0 += 1
                elif winner == playersList[1]:
                    numberOfWinsForPlayer1 += 1
                elif winner == 'draw':
                    numberOfDraws += 1
                else:
                    raise ValueError("Unknown winner '{}'".format(winner))
                #print ("positionsList = \n{}\nwinner = {}".format(positionsList, winner))
            logging.info("winRateForPlayer0 = {}; winRateForPlayer1 = {}; drawRate = {}".format(numberOfWinsForPlayer0/numberOfSimulations, numberOfWinsForPlayer1/numberOfSimulations, numberOfDraws/numberOfSimulations))

    else:
        raise NotImplementedError ("main(): Unknown test type '{}'".format(args.testToDo))


if __name__ == '__main__':
    main()