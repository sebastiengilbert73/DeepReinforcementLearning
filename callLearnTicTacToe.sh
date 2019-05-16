#! /bin/bash
python learnTicTacToe.py \
	--learningRate=0.003 \
	--numberOfEpochs=1000 \
	--minibatchSize=16 \
	--outputDirectory=outputs/ \
	--proportionOfRandomInitialPositions=0.5 \
	--maximumNumberOfMovesForInitialPositions=8 \
	--numberOfInitialPositions=128 \
	--numberOfGamesForEvaluation=11 \
	--weightForTheValueLoss=0.0 \
	--numberOfStandardDeviationsBelowAverageForValueEstimate=0.0 \
	--softMaxTemperatureForSelfPlayEvaluation=1.0 \
	--averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic='{0.4: 1.0, 0.3: 0.3, 0.2: 0.1, 0.1: 0.1}' \
	--epsilon=0.3 \
	--depthOfExhaustiveSearch=1 \



