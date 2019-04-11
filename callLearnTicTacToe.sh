#! /bin/bash
python learnTicTacToe.py \
	--learningRate=0.001 \
	--numberOfEpochs=1000 \
	--minibatchSize=16 \
	--outputDirectory=outputs/ \
	--proportionOfRandomInitialPositions=0.5 \
	--maximumNumberOfMovesForInitialPositions=8 \
	--numberOfInitialPositions=256 \
	--numberOfGamesForEvaluation=11 \
	--weightForTheValueLoss=0.1 \
	--softMaxTemperatureForSelfPlayEvaluation=2.0 \
	--averageTrainingLossToSoftMaxTemperatureForSelfPlayEvaluationDic='{0.4: 1.0, 0.3: 0.5, 0.2: 0.3, 0.1: 0.2}' \


