#! /bin/bash
python learnConnect4.py \
	--learningRate=0.001 \
	--numberOfEpochs=1000 \
	--minibatchSize=16 \
	--outputDirectory='outputs' \
	--proportionOfRandomInitialPositions=1.0 \
	--maximumNumberOfMovesForInitialPositions=40 \
	--numberOfInitialPositions=64 \
	--numberOfGamesForEvaluation=31 \
	--learningRateExponentialDecay=0.999 \
	--softMaxTemperatureForSelfPlayEvaluation=1.0 \
	--epsilon=0.3 \
	--depthOfExhaustiveSearch=1 \
	--chooseHighestProbabilityIfAtLeast=0.5 \
	--startWithNeuralNetwork='/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_18.pth' \

