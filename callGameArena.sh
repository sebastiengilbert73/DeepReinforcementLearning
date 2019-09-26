#! /bin/bash
python gameArena.py \
	connect4 \
	'[(5, 32), (5, 32), (5, 32)]' \
	--neuralNetwork='/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_356.pth' \
	--numberOfGamesForMoveEvaluation=5 \
	--softMaxTemperature=0.3 \
	--displayExpectedMoveValues \
	--depthOfExhaustiveSearch=2 \
	--opponentPlaysFirst \
	

