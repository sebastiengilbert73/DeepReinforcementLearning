#! /bin/bash
python gameArena.py \
	checkers \
	'[(5, 32), (5, 32), (5, 32)]' \
	--neuralNetwork='/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(6,1,8,8)_[(5,32),(5,32),(5,32)]_(4,1,8,8)_checkers_48.pth' \
	--numberOfGamesForMoveEvaluation=5 \
	--softMaxTemperature=0.3 \
	--displayExpectedMoveValues \
	--depthOfExhaustiveSearch=3 \
	--numberOfTopMovesToDevelop=7 \
	--opponentPlaysFirst

