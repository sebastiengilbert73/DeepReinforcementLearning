#! /bin/bash
python learnCheckers.py \
	--learningRate=0.001 \
	--numberOfEpochs=1000 \
	--minibatchSize=16 \
	--outputDirectory='outputs' \
	--proportionOfRandomInitialPositions=0.5 \
	--maximumNumberOfMovesForInitialPositions=100 \
	--numberOfInitialPositions=32 \
	--numberOfGamesForEvaluation=31 \
	--learningRateExponentialDecay=0.99 \
	--softMaxTemperatureForSelfPlayEvaluation=0.3 \
	--epsilon=0.1 \
	--depthOfExhaustiveSearch=1 \
	--chooseHighestProbabilityIfAtLeast=1.0 \
	--numberOfProcesses=4 \

