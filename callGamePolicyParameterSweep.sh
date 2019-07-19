python gamePolicyParameterSweep.py \
	'/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(2,1,3,3)_[(3,16),(3,16),(3,16)]_(1,1,3,3)_tictactoe_45.pth' \
	'tictactoe' \
	--numberOfGamesPerCell=100 \
	--sweepParameter='numberOfTopMovesToDevelop' \
	--parameterSweptValues='[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' \
	--baselineParameters='{"softMaxTemperature": 0.06, "chooseHighestProbabilityIfAtLeast": 0.5, "numberOfGamesPerActionEvaluation": 31, "moveChoiceMode": "SemiExhaustiveMiniMax", "depthOfExhaustiveSearch": 2, "numberOfTopMovesToDevelop": 2}' \
	
