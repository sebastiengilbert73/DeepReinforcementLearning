#! /bin/bash
python learnTicTacToe.py \
	--learningRate=0.001 \
	--numberOfEpochs=1000 \
	--minibatchSize=16 \
	--outputDirectory='outputs' \
	--proportionOfRandomInitialPositions=0.5 \
	--maximumNumberOfMovesForInitialPositions=8 \
	--numberOfInitialPositions=128 \
	--numberOfGamesForEvaluation=31 \
	--learningRateExponentialDecay=0.99 \
	--weightForTheValueLoss=0.0 \
	--numberOfStandardDeviationsBelowAverageForValueEstimate=0.0 \
	--softMaxTemperatureForSelfPlayEvaluation=0.3 \
	--averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic='{0.3: 0.3, 0.2: 0.3, 0.15: 0.3}' \
	--epsilon=0.1 \
	--depthOfExhaustiveSearch=1 \
	--chooseHighestProbabilityIfAtLeast=1.0 \
	--numberOfProcesses=4 \




