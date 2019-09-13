#!/bin/bash
python neuralNetworksTournament.py \
	connect4 \
	'/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_356.pth' \
	'/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_lossRate0.0067.pth' \
	--maximumDepthOfSemiExhaustiveSearch=2 \
	--numberOfTopMovesToDevelop=3 \
	--softMaxTemperature=0.1 \
	--numberOfGames=600



