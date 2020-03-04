import abc
import numpy
import random
import utilities
import pickle
import torch


class Comparator(abc.ABC):
    """
    Abstract class that compares two positions, when it is the opponent's turn
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def BestPosition(self, position0, position1):
        pass  # return the best position

    def Save(self, filepath):
        binary_file = open(filepath, mode='wb')
        pickle.dump(self, binary_file)
        binary_file.close()


def Load(filepath):
    pickle_in = open(filepath, 'rb')
    evaluator = pickle.load(pickle_in)
    pickle_in.close()
    return evaluator

def TournamentWinner(comparator, positionsList):
    if len(positionsList) < 1:
        raise ValueError("TournamentWinner(): The list of positions is empty")
    #remainingPositionsList = random.sample(positionsList, len(positionsList))
    remainingPositionsList = positionsList
    while (len(remainingPositionsList) > 1):
        # Shuffle the positions
        #remainingPositionsList = random.sample(remainingPositionsList, len(remainingPositionsList))
        numberOfWholePairs = int(len(remainingPositionsList)/2)
        winnersList = []
        for duelNdx in range(numberOfWholePairs):
            winner = comparator.BestPosition(remainingPositionsList[2 * duelNdx], remainingPositionsList[2 * duelNdx + 1])
            winnersList.append(winner)
        if len(remainingPositionsList) %2 == 1: # Give the last position a free pass to the next round
            winnersList.append(remainingPositionsList[-1])
        remainingPositionsList = winnersList
    return remainingPositionsList[0] # Return the last surviving position

def TournamentWinnerWithPositionWinnerPairs(comparator, positionWinnerPairsList):
    positionsList = []
    positionToWinnerDict = {}
    for positionWinnerPair in positionWinnerPairsList:
        positionsList.append(positionWinnerPair[0])
        positionToWinnerDict[positionWinnerPair[0]] = positionWinnerPair[1]
    winner = TournamentWinner(comparator, positionsList)
    return (winner, positionToWinnerDict[winner])

def SimulateAGame(comparator, gameAuthority, startingPosition=None, nextPlayer=None, playerToEpsilonDict=None):
    playersList = gameAuthority.PlayersList()
    if startingPosition is None:
        startingPosition = gameAuthority.InitialPosition()
    if nextPlayer is None:
        nextPlayer = playersList[0]

    moveTensorShape = gameAuthority.MoveTensorShape()
    positionsList = [startingPosition]
    currentPosition = startingPosition
    winner = None
    while winner is None:
        #print ("SimulateAGame(): nextPlayer = {}".format(nextPlayer))
        if nextPlayer == playersList[1]:
            currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])
            #print("SimulateAGame(): playersList[1] turn!")
        randomNbr = numpy.random.rand()
        if playerToEpsilonDict is not None:
            epsilon = playerToEpsilonDict[nextPlayer]
        else:
            epsilon = 0
        if randomNbr < epsilon:
            chosenMoveTensor = utilities.ChooseARandomMove(currentPosition, playersList[0], gameAuthority)
            currentPosition, winner = gameAuthority.Move(currentPosition, playersList[0], chosenMoveTensor)
        else:
            legalCandidatePositionsAfterMoveList = utilities.LegalCandidatePositionsAfterMove(gameAuthority, currentPosition, playersList[0])
            chosenPosition = None
            candidatePositionsList = [] # legalCandidatePositionsAfterMoveList contains (position, winner)
            for legalPositionWinnerPair in legalCandidatePositionsAfterMoveList:
                candidatePositionsList.append(legalPositionWinnerPair[0])
                if legalPositionWinnerPair[1] == playersList[0]:
                    chosenPosition = legalPositionWinnerPair[0]
            if chosenPosition is None:
                chosenPosition = TournamentWinner(comparator, candidatePositionsList)
                #print ("Comparison.SimulateAGame(): chosenPosition = {}".format(chosenPosition))

            for positionWinnerPair in legalCandidatePositionsAfterMoveList:
                if chosenPosition is positionWinnerPair[0]:
                    winner = positionWinnerPair[1]

            currentPosition = chosenPosition

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

def SimulateGamesAgainstARandomPlayer(comparator, gameAuthority, numberOfGames, gameToRewardDict=None):
    playersList = gameAuthority.PlayersList()
    #moveTensorShape = gameAuthority.MoveTensorShape()
    numberOfWinsForComparator = 0
    numberOfWinsForRandomPlayer = 0
    numberOfDraws = 0
    if gameToRewardDict is None:
        gameToRewardDict = {}

    for gameNdx in range(numberOfGames):
        comparatorPlayer = playersList[gameNdx % 2]

        winner = None
        currentPosition = gameAuthority.InitialPosition()
        moveNdx = 0
        positionsList = []
        positionsList.append(currentPosition)
        while winner is None:
            nextPlayer = playersList[moveNdx % 2]
            chosenMoveTensor = None
            if nextPlayer == comparatorPlayer:
                if nextPlayer == playersList[1]:
                    currentPosition = gameAuthority.SwapPositions(currentPosition, playersList[0], playersList[1])

                #legalMovesList = utilities.LegalMoveTensorsList(gameAuthority, currentPosition, playersList[0])
                positionAfterMoveWinnerPairList = utilities.LegalCandidatePositionsAfterMove(gameAuthority, currentPosition, playersList[0])
                candidatePositionsList = [positionWinnerPair[0] for positionWinnerPair in positionAfterMoveWinnerPairList]
                chosenPosition = TournamentWinner(comparator, candidatePositionsList)
                for positionAfterMoveWinnerPair in positionAfterMoveWinnerPairList:
                    if positionAfterMoveWinnerPair[0] is chosenPosition:
                        winner = positionAfterMoveWinnerPair[1]

                currentPosition = chosenPosition
                #currentPosition, winner = gameAuthority.Move(currentPosition, playersList[0], chosenMoveTensor)
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
        if winner == comparatorPlayer:
            numberOfWinsForComparator += 1
            gameToRewardDict[tuple(positionsList)] = 1.0
        elif winner == 'draw':
            numberOfDraws += 1
            gameToRewardDict[tuple(positionsList)] = 0.0
        else:
            numberOfWinsForRandomPlayer += 1
            gameToRewardDict[tuple(positionsList)] = -1.0

    return (numberOfWinsForComparator, numberOfWinsForRandomPlayer, numberOfDraws)

def SimulateRandomGames(authority, minimumNumberOfMovesForInitialPositions, maximumNumberOfMovesForInitialPositions,
                        numberOfPositions, swapIfOddNumberOfMoves=False):
    playersList = authority.PlayersList()
    selectedPositionsList = []
    while len(selectedPositionsList) < numberOfPositions:
        gamePositionsList, winner = SimulateAGame(
            comparator=None, gameAuthority=authority, startingPosition=None, nextPlayer=None,
            playerToEpsilonDict={playersList[0]: 1.0, playersList[1]: 1.0}) # With epsilon=1.0, the comparator will never be called
        if len(gamePositionsList) >= minimumNumberOfMovesForInitialPositions:
            maxNdx = min(maximumNumberOfMovesForInitialPositions - 1, len(gamePositionsList) - 2) # The last index cannot be the last position, since the game is over
            if maxNdx < 0:
                maxNdx = 0
            selectedNdx = numpy.random.randint(minimumNumberOfMovesForInitialPositions, maxNdx + 1) # maxNdx can be selected: {0, 1, 2, ..., maxNdx}
            if swapIfOddNumberOfMoves and selectedNdx %2 == 1:
                selectedPositionsList.append(authority.SwapPositions( gamePositionsList[selectedNdx], playersList[0], playersList[1] ))
            else:
                selectedPositionsList.append(gamePositionsList[selectedNdx])
    return selectedPositionsList

def ComparePositionPairs(authority, comparator, positionTsrList, numberOfGames, epsilon=1.0, playerToEpsilonDict=None):
    if len(positionTsrList) %2 == 1: # If an odd number of positions, ignore the last one
        positionTsrList = positionTsrList[0: -1]
    playersList = authority.PlayersList()
    if playerToEpsilonDict is None:
        playerToEpsilonDict = {playersList[0]: epsilon, playersList[1]: epsilon}
    pairWinnerIndexList = []
    for pairNdx in range(len(positionTsrList)//2):
        position0 = positionTsrList[2 * pairNdx]
        position1 = positionTsrList[2 * pairNdx + 1]
        position0NbrOfWins = 0
        position0NbrOfDraws = 0
        position0NbrOfLosses = 0
        position1NbrOfWins = 0
        position1NbrOfDraws = 0
        position1NbrOfLosses = 0
        for gameNdx in range(numberOfGames):
            positionsList0, winner0 = SimulateAGame(comparator, authority, position0,
                                                    playersList[1], playerToEpsilonDict)
            positionsList1, winner1 = SimulateAGame(comparator, authority, position1,
                                                    playersList[1], playerToEpsilonDict)
            #print ("ComparePositionPairs(): positionsList0: \n{}".format(positionsList0))
            #print ("ComparePositionPairs(): positionsList1: \n{}".format(positionsList1))
            if winner0 == playersList[0]:
                position0NbrOfWins += 1
            elif winner0 == 'draw':
                position0NbrOfDraws += 1
            elif winner0 == playersList[1]:
                position0NbrOfLosses += 1
            else:
                raise ValueError("ComparePositionPairs(): Unknown winner0: {}".format(winner0))
            if winner1 == playersList[0]:
                position1NbrOfWins += 1
            elif winner1 == 'draw':
                position1NbrOfDraws += 1
            elif winner1 == playersList[1]:
                position1NbrOfLosses += 1
            else:
                raise ValueError("ComparePositionPairs(): Unknown winner1: {}".format(winner1))
        expectedReward0 = (position0NbrOfWins - position0NbrOfLosses)/numberOfGames
        expectedReward1 = (position1NbrOfWins - position1NbrOfLosses) / numberOfGames
        #print ("ComparePositionPairs(): expectedReward0 = {}; expectedReward1 = {}".format(expectedReward0, expectedReward1))
        if expectedReward0 >= expectedReward1:
            pairWinnerIndexList.append(0)
        else:
            pairWinnerIndexList.append(1)
    return pairWinnerIndexList

class IntegerComparator(Comparator):
    def __init__(self):
        pass

    def BestPosition(self, int0, int1):
        if int0 > int1:
            return int0
        else:
            return int1

class ComparatorsEnsemble(Comparator):
    def __init__(self,
                 comparatorsList):
        self.comparatorsList = comparatorsList

    def BestPosition(self, position0, position1):
        votesForPosition0 = 0
        votesForPosition1 = 0
        for comparator in self.comparatorsList:
            comparators_bestPosition = comparator.BestPosition(position0, position1)
            if torch.equal(comparators_bestPosition, position0):
                votesForPosition0 += 1
            elif torch.equal(comparators_bestPosition, position1):
                votesForPosition1 += 1
            else:
                raise ValueError("ComparatorsEnsemble.BestPosition(): The best position returned by a comparator ({}) is not position0 nor position1".format(comparators_bestPosition))
        if votesForPosition0 > votesForPosition1:
            return position0
        elif votesForPosition1 > votesForPosition0:
            return position1
        else: # Draw
            if numpy.random.random() < 0.5:
                return position0
            else:
                return position1



def main():
    print ("Comparison.py main()")
    import tictactoe
    import ComparisonNet
    import autoencoder

    authority = tictactoe.Authority()
    autoencoderNet = autoencoder.position.Net()
    autoencoderNet.Load('/home/sebastien/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(2,1,3,3)_[(3,64,1)]_32_noZeroPadding_tictactoeAutoencoder_1000.pth')

    comparator = ComparisonNet.BuildADecoderClassifierFromAnAutoencoder(
        autoencoderNet)

    position0 = authority.InitialPosition()
    position0[0, 0, 0, 1] = 1
    position1 = authority.InitialPosition()
    position1[0, 0, 1, 1] = 1
    position2 = authority.InitialPosition()
    position2[0, 0, 0, 0] = 1
    position3 = authority.InitialPosition()
    position3[0, 0, 1, 0] = 1
    positionsList = [position0, position1, position2, position3]

    pairWinnerIndexList = ComparePositionPairs(authority, comparator, positionsList, numberOfGames=20, epsilon=0.1)
    print ("pairWinnerIndexList = {}".format(pairWinnerIndexList))


if __name__ == '__main__':
    main()