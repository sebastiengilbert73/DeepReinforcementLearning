#! /bin/bash
python learnConnect4.py \
	--learningRate=0.0001 \
	--numberOfEpochs=1000 \
	--minibatchSize=16 \
	--outputDirectory='outputs' \
	--proportionOfRandomInitialPositions=1.0 \
	--maximumNumberOfMovesForInitialPositions=40 \
	--numberOfInitialPositions=64 \
	--numberOfGamesForEvaluation=31 \
	--learningRateExponentialDecay=0.999 \
	--softMaxTemperatureForSelfPlayEvaluation=0.3 \
	--epsilon=0.1 \
	--depthOfExhaustiveSearch=1 \
	--chooseHighestProbabilityIfAtLeast=0.0 \

