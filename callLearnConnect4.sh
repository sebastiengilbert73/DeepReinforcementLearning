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

