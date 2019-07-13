import argparse
import ast
import moveEvaluation.ConvolutionStack
import tictactoe
import policy
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('neuralNetwork', help='Filepath to the neural network')
parser.add_argument('game', help='The name of the game')
parser.add_argument('--numberOfGamesPerCell', help='The number of games played per parameter cell. Default: 300', type=int, default=300)
parser.add_argument('--sweepParameter', help='The parameter whose value will be swept. Default: softMaxTemperature', default='softMaxTemperature')
parser.add_argument('--parameterSweptValues', help="The list of parameter values. Default: '[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]'", default='[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]')
parser.add_argument('--baselineParameters', help="The baseline parameter values, in a dictionary (the sweep parameter value will be ignored). Default: {'softMaxTemperature': 0.1, 'chooseHighestProbabilityIfAtLeast': 0.3, 'numberOfGamesPerActionEvaluation': 31, 'moveChoiceMode': 'SoftMax', 'depthOfExhaustiveSearch': 2, 'numberOfTopMovesToDevelop': 3}", \
    default='{"softMaxTemperature": 0.1, "chooseHighestProbabilityIfAtLeast": 0.3, "numberOfGamesPerActionEvaluation": 31, "moveChoiceMode": "SoftMax", "depthOfExhaustiveSearch": 2, "numberOfTopMovesToDevelop": 3}')
parser.add_argument('--outputFile', help='The output file. Default: ./output.csv', default='./outputs/parameterSweep.csv')
args = parser.parse_args()


def main():
    print ("gamePolicyParameterSweep.py main()")
    parameterSweptValuesList = ast.literal_eval(args.parameterSweptValues)
    parametersDic = ast.literal_eval(args.baselineParameters)
    #print ("main(): parametersDic = {}".format(parametersDic))

    # Load the neural network
    neuralNetwork = moveEvaluation.ConvolutionStack.Net()
    neuralNetwork.Load(args.neuralNetwork)

    # Load the game authority
    if args.game == 'tictactoe':
        authority = tictactoe.Authority()
    else:
        raise NotImplementedError("main(): The game '{}' is not implemented".format(args.game))
    playersList = authority.PlayersList()

    # Output monitoring file
    outputFile = open(args.outputFile, "w",
                         buffering=1)  # Flush the buffer at each line
    outputFile.write(
        "{},averageTime,winRate,drawRate,lossRate\n".format(args.sweepParameter))


    for sweptValue in parameterSweptValuesList:
        print ("main() {} = {}".format(args.sweepParameter, sweptValue))
        if args.sweepParameter not in parametersDic:
            raise ValueError("main(): The sweep parameter '{}' is not in the dictionary".format(args.sweepParameter))
        parametersDic[args.sweepParameter] = sweptValue
        """if args.sweepParameter == 'softMaxTemperature':
            parametersDic['softMaxTemperature'] = sweptValue
        else:
            raise NotImplementedError("main(): The sweep parameter '{}' is not implemented".format(args.sweepParameter))
        """
        startTime = time.time()
        (averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate, losingGamePositionsListList) = \
            policy.AverageRewardAgainstARandomPlayerKeepLosingGames(
                playersList,
                authority,
                neuralNetwork,
                chooseHighestProbabilityIfAtLeast=parametersDic['chooseHighestProbabilityIfAtLeast'],
                preApplySoftMax=True,
                softMaxTemperature=parametersDic['softMaxTemperature'],
                numberOfGames=args.numberOfGamesPerCell,
                moveChoiceMode=parametersDic['moveChoiceMode'],
                numberOfGamesForMoveEvaluation=parametersDic['numberOfGamesPerActionEvaluation'],  # ignored by SoftMax
                depthOfExhaustiveSearch=parametersDic['depthOfExhaustiveSearch'],
                numberOfTopMovesToDevelop=parametersDic['numberOfTopMovesToDevelop']
            )
        endTime = time.time()
        averageTime = (endTime - startTime)/args.numberOfGamesPerCell
        print ("main(): averageTime = {}; winRate = {}; drawRate = {}; lossRate = {}".format(averageTime, winRate, drawRate, lossRate))
        outputFile.write(str(sweptValue) + ',' + str(averageTime) + ',' + str(winRate) + ',' \
            + str(drawRate) + ',' + str(lossRate) + '\n')
if __name__ == '__main__':
    main()
