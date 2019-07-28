python gamePolicyParameterSweep.py \
	'/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(2,1,3,3)_[(3,32),(3,32),(3,32)]_(1,1,3,3)_tictactoe_97.pth' \
	'tictactoe' \
	--numberOfGamesPerCell=100 \
	--sweepParameter='numberOfTopMovesToDevelop' \
	--parameterSweptValues='[1, 2, 3, 4, 5, 6, 7, 8, 9]' \
	--baselineParameters='{"softMaxTemperature": 0.06, "chooseHighestProbabilityIfAtLeast": 1.0, "numberOfGamesPerActionEvaluation": 31, "moveChoiceMode": "SemiExhaustiveMiniMax", "depthOfExhaustiveSearch": 3, "numberOfTopMovesToDevelop": 3}' \
	
