#!/bin/bash
python netEnsemblesTournament.py \
	connect4 \
	'["/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_356.pth"]' \
	'["/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_356.pth", 
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033b.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033c.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033d.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033e.pth",
	"/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.pth"]' \
	--maximumDepthOfSemiExhaustiveSearch=3 \
	--numberOfTopMovesToDevelop=3 \
	--softMaxTemperature=0.1 \
	--numberOfGames=10 \
	--displayPositions
