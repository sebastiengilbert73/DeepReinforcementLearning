import torch
import logging
import collections # OrderedDict
import utilities
import pickle
import numpy
import abc

class Regressor(abc.ABC):
    """
    Abstract class that predicts the win rates, when it is the opponent's turn
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def WinRates(self, positionEncoding):
        pass  # Predict (winRate0, drawRate, winRate1)

    def Save(self, filepath):
        binary_file = open(filepath, mode='wb')
        pickle.dump(self, binary_file)
        binary_file.close()


class Net(Regressor,
          torch.nn.Module):
    def __init__(self,
                 inputNumberOfAttributes=16,
                 bodyStructureList=[8, 4, 3],
                 dropoutRatio=0.1,
                 ):
        super(Net, self).__init__()
        self.inputNumberOfAttributes = inputNumberOfAttributes
        self.bodyStructureList = bodyStructureList
        self.dropoutRatio = dropoutRatio

        bodyStructureDict = collections.OrderedDict()

        for layerNdx in range(len(bodyStructureList) - 1):
            layerName = 'layer_' + str(layerNdx)
            if layerNdx == 0:
                numberOfInputs = self.inputNumberOfAttributes
            else:
                numberOfInputs = self.bodyStructureList[layerNdx - 1]
            numberOfOutputs = self.bodyStructureList[layerNdx]

            bodyStructureDict[layerName] = torch.nn.Sequential(
                torch.nn.Linear(numberOfInputs, numberOfOutputs),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dropoutRatio)
            )
        # Last layer
        bodyStructureDict['layer_' + str(len(bodyStructureList) - 1)] = \
            torch.nn.Sequential(
                torch.nn.Linear(self.bodyStructureList[-2], self.bodyStructureList[-1])
            )


        self.layers = torch.nn.Sequential(bodyStructureDict)

    def forward(self, inputTsr):
        outputTsr = self.layers(inputTsr)
        clampedOutputTsr = torch.nn.functional.relu(outputTsr)
        # Normalize to a sum of 1.0 along the rows
        clampedOutputTsr = torch.nn.functional.normalize(clampedOutputTsr, p=1, dim=1)
        return clampedOutputTsr

    def WinRates(self, positionEncoding):
        outputTsr = self.forward(positionEncoding.unsqueeze(0))
        winRate0 = outputTsr[0][0].item()
        drawRate = outputTsr[0][1].item()
        winRate1 = outputTsr[0][2].item()
        return (winRate0, drawRate, winRate1)





def Load(filepath):
    pickle_in = open(filepath, 'rb')
    regressor = pickle.load(pickle_in)
    pickle_in.close()
    return regressor

def SimulateGamesAgainstARandomPlayer(regressor, encoder, gameAuthority, numberOfGames, gameToRewardDict=None):
    playersList = gameAuthority.PlayersList()
    numberOfWinsForRegressor = 0
    numberOfWinsForRandomPlayer = 0
    numberOfDraws = 0
    if gameToRewardDict is None:
        gameToRewardDict = {}

    for gameNdx in range(numberOfGames):
        regressorPlayer = playersList[gameNdx % 2]

        winner = None
        currentPosition = gameAuthority.InitialPosition()
        moveNdx = 0
        positionsList = []
        positionsList.append(currentPosition)
        while winner is None:
            nextPlayer = playersList[moveNdx % 2]
            chosenMoveTensor = None
            if nextPlayer == regressorPlayer:
                if nextPlayer == playersList[1]:
                    currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])

                #legalMovesList = utilities.LegalMoveTensorsList(gameAuthority, currentPosition, playersList[0])
                positionAfterMoveToWinnerDict = utilities.LegalCandidatePositionsAfterMoveDictionary(gameAuthority, currentPosition, playersList[0])
                #candidatePositionsList = [positionWinnerPair[0] for positionWinnerPair in positionAfterMoveWinnerPairList]
                highestReward = -2.0
                chosenPosition = None
                for candidatePosition, candidateWinner in positionAfterMoveToWinnerDict.items():
                    encoding = encoder.Encode(candidatePosition.unsqueeze(0))
                    #encodingOutputTsr = regressor(encoding.unsqueeze(0)).squeeze()
                    #winRate, drawRate, lossRate = encodingOutputTsr[0], encodingOutputTsr[1], encodingOutputTsr[2]
                    (winRate, drawRate, lossRate) = regressor.WinRates(encoding.squeeze())
                    reward = winRate - lossRate
                    if candidateWinner == playersList[0] or (candidateWinner is not playersList[0] and reward > highestReward):
                        highestReward = reward
                        chosenPosition = candidatePosition
                if chosenPosition is None:
                    raise ValueError("SimulateGamesAgainstARandomPlayer(): chosenPosition is None... positionAfterMoveToWinnerDict = {}".format(positionAfterMoveToWinnerDict))
                currentPosition = chosenPosition
                winner = positionAfterMoveToWinnerDict[chosenPosition]

                if nextPlayer == playersList[1]:  # De-swap, reverse the winner
                    currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
                    if winner == playersList[0]:
                        winner = playersList[1]
                    elif winner == playersList[1]:
                        winner = playersList[0]
                positionsList.append(currentPosition)

            else: # Random player's turn
                chosenMoveTensor = utilities.ChooseARandomMove(currentPosition, nextPlayer, gameAuthority)
                currentPosition, winner = gameAuthority.Move(currentPosition, nextPlayer, chosenMoveTensor)
                positionsList.append(currentPosition)

            moveNdx += 1
        if winner == regressorPlayer:
            numberOfWinsForRegressor += 1
            gameToRewardDict[tuple(positionsList)] = 1.0
        elif winner == 'draw':
            numberOfDraws += 1
            gameToRewardDict[tuple(positionsList)] = 0.0
        else:
            numberOfWinsForRandomPlayer += 1
            gameToRewardDict[tuple(positionsList)] = -1.0

    return (numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws)

def SimulateAGame(regressor, encoder, gameAuthority, startingPosition=None, nextPlayer=None, playerToEpsilonDict=None):
    playersList = gameAuthority.PlayersList()
    if startingPosition is None:
        startingPosition = gameAuthority.InitialPosition()
    if nextPlayer is None:
        nextPlayer = playersList[0]

    positionsList = [startingPosition]
    currentPosition = startingPosition
    winner = None
    while winner is None:
        if nextPlayer == playersList[1]:
            currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
        randomNbr = numpy.random.rand()
        if playerToEpsilonDict is not None:
            epsilon = playerToEpsilonDict[nextPlayer]
        else:
            epsilon = 0
        if randomNbr < epsilon:
            chosenMoveTensor = utilities.ChooseARandomMove(currentPosition, playersList[0], gameAuthority)
            currentPosition, winner = gameAuthority.Move(currentPosition, playersList[0], chosenMoveTensor)
        else:
            positionAfterMoveToWinnerDict = utilities.LegalCandidatePositionsAfterMoveDictionary(gameAuthority,
                                                                                                 currentPosition,
                                                                                                 playersList[0])
            highestReward = -2.0
            chosenPosition = None
            for candidatePosition, candidateWinner in positionAfterMoveToWinnerDict.items():
                """candidatePositionsList.append(legalPositionWinnerPair[0])
                if legalPositionWinnerPair[1] == playersList[0]:
                    chosenPosition = legalPositionWinnerPair[0]
                """
                encoding = encoder.Encode(candidatePosition.unsqueeze(0))
                #encodingOutputTsr = regressor(encoding.unsqueeze(0)).squeeze()
                #winRate, drawRate, lossRate = encodingOutputTsr[0], encodingOutputTsr[1], encodingOutputTsr[2]
                (winRate, drawRate, lossRate) = regressor.WinRates(encoding.squeeze())
                reward = winRate - lossRate
                if candidateWinner == playersList[0] or (candidateWinner is not playersList[0] and reward > highestReward):
                    highestReward = reward
                    chosenPosition = candidatePosition

            currentPosition = chosenPosition
            winner = positionAfterMoveToWinnerDict[chosenPosition]

        if nextPlayer == playersList[1]: # De-swap, reverse the winner
            currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
            #print ("SimulateAGame(): playersList[1]: Swapped position after move: \n{}".format(currentPosition))
            if winner == playersList[0]:
                winner = playersList[1]
            elif winner == playersList[1]:
                winner = playersList[0]

        positionsList.append(currentPosition)

        if nextPlayer == playersList[0]:
            nextPlayer = playersList[1]
        else:
            nextPlayer = playersList[0]
    return positionsList, winner



if __name__ == '__main__':
    print("winRatesRegression.py __main__")
    regressor = Net(10, [8, 4, 3], dropoutRatio=0.1)
    print ("regressor = {}".format(regressor))

    inputTsr = torch.randn(14, 10)
    outputTsr = regressor(inputTsr)
    print ("outputTsr = {}".format(outputTsr))
    print ("outputTsr.shape = {}".format(outputTsr.shape))

    import tictactoe
    import autoencoder.position
    authority = tictactoe.Authority()
    encoder = autoencoder.position.Net()
    encoder.Load('/home/sebastien/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(2,1,3,3)_[(2,128,1),(2,128,1)]_10_noZeroPadding_tictactoeAutoencoder_115.pth')
    (numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws) = SimulateGamesAgainstARandomPlayer(regressor, encoder, authority, 100, None)
    print ("numberOfWinsForRegressor = {}; numberOfWinsForRandomPlayer = {}; numberOfDraws = {}".format(numberOfWinsForRegressor, numberOfWinsForRandomPlayer, numberOfDraws))
