import argparse
import numpy
import torch
import os
import ast
import tictactoe
import logging
import autoencoder.position # autoencoder
import winRatesRegression

parser = argparse.ArgumentParser()
parser.add_argument('regressorEnsembleFilepath', help='The prefix of the filepath of the regressor ensemble')
parser.add_argument('outputFilepath', help='The csv file where the results will be written')
parser.add_argument('autoencoderFilepath', help='The filepath of the autoencoder')
parser.add_argument('netEnsembleFilepath', help='The filepath to save the ensemble')
parser.add_argument('--numberOfSimulationsPerPosition', help='For each starting position, the number of simulations. Default: 64', type=int, default=64)
parser.add_argument('--epsilon', help='Epsilon (probability of a random move) for the simulations. Default: 0.5', type=float, default=0.5)
parser.add_argument('--numberOfPositions', help='The number of positions to evaluate. Default: 1000', type=int, default=1000)
parser.add_argument('--preAssembledEnsembleFilepath', help="If you want to use an already assembled ensemble (fo example, after filtering it), define this filepath. Default: 'None'", default='None')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')


def main():
    logging.info("regressorCreatesSamples.py main()")

    authority = tictactoe.Authority()
    #positionTsrShape = authority.PositionTensorShape()
    playersList = authority.PlayersList()

    # Load the ensemble
    if args.epsilon < 1.0:
        if args.preAssembledEnsembleFilepath == 'None':
            neuralNetworksDirectory = os.path.dirname(args.regressorEnsembleFilepath)
            #print ("neuralNetworksDirectory = {}".format(neuralNetworksDirectory))
            neuralNetworksFilenames = \
                [filename for filename in os.listdir(neuralNetworksDirectory ) if filename.startswith( os.path.basename( args.regressorEnsembleFilepath) )]
            #print ("neuralNetworksFilepaths = {}".format(neuralNetworksFilepaths))
            neuralNetworksFilepaths = [os.path.join(neuralNetworksDirectory, filename) for filename in neuralNetworksFilenames]
            #print ("neuralNetworksFilepaths = {}".format(neuralNetworksFilepaths))

            neuralNetworksList = []
            for filepath in neuralNetworksFilepaths:
                neuralNet = winRatesRegression.Load(filepath)
                neuralNetworksList.append(neuralNet)
            netEnsemble = winRatesRegression.RegressorsEnsemble(neuralNetworksList)
            netEnsemble.Save(args.netEnsembleFilepath)
        else:
            logging.info("Using pre-assembled ensemble {}".format(args.preAssembledEnsembleFilepath))
            netEnsemble = winRatesRegression.Load(args.preAssembledEnsembleFilepath)
    else:
        netEnsemble = None

    # Create the autoencoder
    encoder = autoencoder.position.Net()
    encoder.Load(args.autoencoderFilepath)
    numberOfLatentVariables = encoder.numberOfLatentVariables
    header = ''
    for latentNdx in range(numberOfLatentVariables):
        header += 'p' + str(latentNdx) + ','

    # Create the output file
    outputFile = open(args.outputFilepath, "w",
                         buffering=1)  # Flush the buffer at each line
    outputFile.write(
        header + "player0WinRate,drawRate,player1WinRate\n")



    for positionNdx in range(1, args.numberOfPositions + 1):
        logging.info("Generating position {}...".format(positionNdx))
        startingPosition = winRatesRegression.SimulateRandomGames(authority,
                                                                    encoder=encoder,
                                                                    minimumNumberOfMovesForInitialPositions=0,
                                                                    maximumNumberOfMovesForInitialPositions=7,
                                                                    numberOfPositions=1,
                                                                    swapIfOddNumberOfMoves=False)[0]
        authority.Display(startingPosition)
        numberOfWinsForPlayer0 = 0
        numberOfWinsForPlayer1 = 0
        numberOfDraws = 0
        for simulationNdx in range(args.numberOfSimulationsPerPosition):
            (positionsList, winner) = winRatesRegression.SimulateAGame(netEnsemble, encoder, authority,
                                                                       startingPosition=startingPosition,
                                                                       nextPlayer=playersList[1],
                                                                       playerToEpsilonDict={playersList[0]: args.epsilon,
                                                                                    playersList[1]: args.epsilon})
            if winner == playersList[0]:
                numberOfWinsForPlayer0 += 1
            elif winner == playersList[1]:
                numberOfWinsForPlayer1 += 1
            elif winner == 'draw':
                numberOfDraws += 1
            else:
                raise ValueError("Unknown winner '{}'".format(winner))
            # print ("positionsList = \n{}\nwinner = {}".format(positionsList, winner))
        player0WinRate = numberOfWinsForPlayer0/args.numberOfSimulationsPerPosition
        player1WinRate = numberOfWinsForPlayer1/args.numberOfSimulationsPerPosition
        drawRate = numberOfDraws/args.numberOfSimulationsPerPosition
        logging.info("winRateForPlayer0 = {}; drawRate = {}; winRateForPlayer1 = {}".format(
            player0WinRate, drawRate, player1WinRate ))

        #positionList = startingPosition.flatten().tolist()
        positionEncoding = encoder.Encode(startingPosition.unsqueeze(0)).flatten().tolist()
        print ("positionEncoding = {}".format(positionEncoding))
        for encodingNdx in range(len(positionEncoding)):
            outputFile.write("{},".format(positionEncoding[encodingNdx]))
        outputFile.write("{},{},{}\n".format(player0WinRate, drawRate, player1WinRate))


if __name__ == '__main__':
    main()