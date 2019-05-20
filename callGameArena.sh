#! /bin/bash
python gameArena.py \
	connect4 \
	'[(5, 16), (5, 16), (5, 16)]' \
	--numberOfGamesForMoveEvaluation=31 \
	--softMaxTemperature=1.0 \
	--displayExpectedMoveValues \
	--depthOfExhaustiveSearch=1

