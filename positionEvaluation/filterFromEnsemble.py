import winRatesRegression # type: ignore
import logging
import argparse
from typing import Dict, Tuple
import tictactoe
import autoencoder.position
import os

parser = argparse.ArgumentParser()
parser.add_argument('ensembleFilepath', help='The filepath to the ensemble regressor')
parser.add_argument('autoencoderFilepath', help='The filepath to the autoencoder')
parser.add_argument('--numberOfGames', help='The number of games played against a random player. Default: 100', type=int, default=100)
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs'", default='./outputs')
parser.add_argument('--fractionOfBestRegressorsToKeep', help="The fraction of the regressors that we'll keep. Default: 0.33", type=float, default=0.33)

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main() -> None:
    logging.info("filterFromEnsemble.py main()")

    # Load the ensemble
    ensemble: winRatesRegression.RegressorsEnsemble = winRatesRegression.Load(args.ensembleFilepath)

    authority: tictactoe.Authority = tictactoe.Authority()

    # Load the encoder
    encoder: autoencoder.position.Net = autoencoder.position.Net()
    encoder.Load(args.autoencoderFilepath)

    neuralNetworkToWinRatesDict: Dict[winRatesRegression.Regressor, Tuple[float, float, float]] = {}
    rewardsList = []
    # Loop through the neural networks
    for neuralNetwork in ensemble.regressorsList:
        (numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws) = winRatesRegression.SimulateGamesAgainstARandomPlayer(
            neuralNetwork, encoder, authority, args.numberOfGames
        )
        logging.info("numberOfWinsForRegressor = {}; numberOfWinsForRandomPlayer = {}; numberOfDraws = {}".format(
            numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws
        ))
        neuralNetworkToWinRatesDict[neuralNetwork] = (numberOfWinsForRegressor/args.numberOfGames,
                                                      numberOfDraws/args.numberOfGames,
                                                      numberOfWinsForRandomPlayer / args.numberOfGames,)

        rewardsList.append((numberOfWinsForRegressor - numberOfWinsForRandomPlayer)/args.numberOfGames)

    # Write the win rates
    winRatesFile = open(os.path.join(args.outputDirectory, 'regressorWinRates.csv'), "w",
                         buffering=1)  # Flush the buffer at each line
    winRatesFile.write(
        "winRateForRegressor,drawRate,winRateForRandomPlayer,reward\n")

    for neuralNet, winRatesTuple in neuralNetworkToWinRatesDict.items():
        winRatesFile.write("{},{},{},{}\n".format(winRatesTuple[0], winRatesTuple[1],
                                               winRatesTuple[2], winRatesTuple[0] - winRatesTuple[2]))

    # Get the best rewards
    rewardsList.sort(reverse=True) # Sort in descending order
    numberOfRegressorsToKeep = round(args.fractionOfBestRegressorsToKeep * len(ensemble.regressorsList))
    logging.info("numberOfRegressorsToKeep = {}".format(numberOfRegressorsToKeep))
    print ("rewardsList = \n{}".format(rewardsList))
    rewardThreshold = rewardsList[numberOfRegressorsToKeep - 1]
    logging.info("rewardThreshold = {}".format(rewardThreshold))

    # Create a new ensemble with the best regressors
    bestRegressorsList = []
    for regressor in ensemble.regressorsList:
        regressorWinRate = neuralNetworkToWinRatesDict[regressor][0]
        regressorLossRate = neuralNetworkToWinRatesDict[regressor][2]
        regressorReward = regressorWinRate - regressorLossRate
        if regressorReward >= rewardThreshold:
            bestRegressorsList.append(regressor)
    logging.info("len(bestRegressorsList) = {}".format(len(bestRegressorsList)))

    eliteEnsemble = winRatesRegression.RegressorsEnsemble(bestRegressorsList)
    eliteEnsemble.Save(os.path.join(args.outputDirectory, "eliteEnsemble.bin"))


if __name__ == '__main__':
    main()