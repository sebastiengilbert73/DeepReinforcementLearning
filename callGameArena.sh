#! /bin/bash
python gameArena.py \
	connect4 \
	'[(5, 32), (5, 32), (5, 32)]' \
	--neuralNetwork='["/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033b.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033c.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033d.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033e.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.pth"]' \
	--numberOfGamesForMoveEvaluation=5 \
	--softMaxTemperature=0.3 \
	--displayExpectedMoveValues \
	--depthOfExhaustiveSearch=3 \
	--numberOfTopMovesToDevelop=7 \
	#--opponentPlaysFirst

