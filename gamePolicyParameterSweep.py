import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('neuralNetwork', help='Filepath to the neural network')
parser.add_argument('game', help='The name of the game')
parser.add_argument('--numberOfGamesPerCell', help='The number of games played per parameter cell. Default: 300', type=int, default=300)
parser.add_argument('--sweepParameter', help='The parameter whose value will be swept. Default: softMaxTemperature', default='softMaxTemperature')
parser.add_argument('--parameterSweptValues', help="The list of parameter values. Default: '[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]'", default='[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]')
parser.add_argument('--baselineParameters', help="The baseline parameter values, in a dictionary (the sweep parameter value will be ignored). Default: {'softMaxTemperature': 0.1, 'chooseHighestProbabilityIfAtLeast': 0.3, 'numberOfGamesPerActionEvaluation': 31, 'moveChoiceMode': 'SoftMax', 'depthOfExhaustiveSearch': 2}", \
    default='{"softMaxTemperature": 0.1, "chooseHighestProbabilityIfAtLeast": 0.3, "numberOfGamesPerActionEvaluation": 31, "moveChoiceMode": "SoftMax", "depthOfExhaustiveSearch": 2}')
args = parser.parse_args()


def main():
    print ("gamePolicyParameterSweep.py main()")
    parameterSweptValuesList = ast.literal_eval(args.parameterSweptValues)
    baselineParametersDic = ast.literal_eval(args.baselineParameters)
    print ("main(): baselineParametersDic = {}".format(baselineParametersDic))

if __name__ == '__main__':
    main()
