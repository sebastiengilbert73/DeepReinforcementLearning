python gamePolicyParameterSweep.py \
	'/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,3,3)_[(3,16),(3,16),(3,16)]_(1,1,3,3)_tictactoe_295.pth' \
	'tictactoe' \
	--numberOfGamesPerCell=100 \
	--sweepParameter='softMaxTemperature' \
	--parameterSweptValues='[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]' \
	--baselineParameters='{"softMaxTemperature": 0.06, "chooseHighestProbabilityIfAtLeast": 0.5, "numberOfGamesPerActionEvaluation": 31, "moveChoiceMode": "SemiExhaustiveSoftMax", "depthOfExhaustiveSearch": 3, "numberOfTopMovesToDevelop": 2}' \
	
