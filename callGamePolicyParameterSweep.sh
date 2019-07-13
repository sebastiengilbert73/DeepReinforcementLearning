python gamePolicyParameterSweep.py \
	'/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,3,3)_[(3,16),(3,16),(3,16)]_(1,1,3,3)_tictactoe_295.pth' \
	'tictactoe' \
	--numberOfGamesPerCell=300 \
	--sweepParameter='chooseHighestProbabilityIfAtLeast' \
	--parameterSweptValues='[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]' \
	--baselineParameters='{"softMaxTemperature": 0.1, "chooseHighestProbabilityIfAtLeast": 0.3, "numberOfGamesPerActionEvaluation": 31, "moveChoiceMode": "SoftMax", "depthOfExhaustiveSearch": 2, "numberOfTopMovesToDevelop": 3}' \
	
