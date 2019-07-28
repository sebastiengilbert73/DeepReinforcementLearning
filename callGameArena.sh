#! /bin/bash
python gameArena.py \
	connect4 \
	'[(5, 32), (5, 32), (5, 32)]' \
	--neuralNetwork='/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_8.pth' \
	--numberOfGamesForMoveEvaluation=31 \
	--softMaxTemperature=0.3 \
	--displayExpectedMoveValues \
	--depthOfExhaustiveSearch=1 \

