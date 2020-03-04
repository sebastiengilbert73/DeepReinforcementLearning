import argparse
import numpy
import torch
import os
import ast
import tictactoe
import logging
import autoencoder.position # autoencoder
import Comparison

parser = argparse.ArgumentParser()
parser.add_argument('comparatorFilepath', help='The prefix to the filepath of the comparator')
parser.add_argument('outputFilepath', help='The csv file where the results will be written')
parser.add_argument('autoencoderFilepath', help='The filepath of the autoencoder')
parser.add_argument('--numberOfSimulationsPerPosition', help='For each starting position, the number of simulations. Default: 64', type=int, default=64)
parser.add_argument('--epsilon', help='Epsilon (probability of a random move) for the simulations. Default: 0.5', type=float, default=0.5)
parser.add_argument('--numberOfPositions', help='The number of positions to evaluate. Default: 1000', type=int, default=1000)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')


def main():
    logging.info("comparatorCreatesSamples.py main()")

    authority = tictactoe.Authority()
    positionTsrShape = authority.PositionTensorShape()
    playersList = authority.PlayersList()

    # Load the ensemble
    netEnsemble = Comparison.Load(args.comparatorFilepath)

    # Create the output file
    outputFile = open(args.outputFilepath, "w",
                         buffering=1)  # Flush the buffer at each line
    outputFile.write(
        "p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,player0WinRate,drawRate,player1WinRate\n")

    # Create the autoencoder
    encoder = autoencoder.position.Net()
    encoder.Load(args.autoencoderFilepath)

    for positionNdx in range(1, args.numberOfPositions + 1):
        logging.info("Generating position {}...".format(positionNdx))
        startingPosition = Comparison.SimulateRandomGames(authority,
                                                               0,
                                                               7,
                                                               1,
                                                               swapIfOddNumberOfMoves=False)[0]
        authority.Display(startingPosition)
        numberOfWinsForPlayer0 = 0
        numberOfWinsForPlayer1 = 0
        numberOfDraws = 0
        for simulationNdx in range(args.numberOfSimulationsPerPosition):
            (positionsList, winner) = Comparison.SimulateAGame(netEnsemble, authority,
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

        positionList = startingPosition.flatten().tolist()
        positionEncoding = encoder.Encode(startingPosition.unsqueeze(0)).flatten().tolist()
        print ("positionEncoding = {}".format(positionEncoding))
        for encodingNdx in range(len(positionEncoding)):
            outputFile.write("{},".format(positionEncoding[encodingNdx]))
        outputFile.write("{},{},{}\n".format(player0WinRate, drawRate, player1WinRate))


if __name__ == '__main__':
    main()