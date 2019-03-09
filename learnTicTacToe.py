import torch
import policy
import tictactoe

playerList = ['X', 'O']

def SimulateGameAndGetReward(positionTensor, nextPlayer, authority, neuralNetwork):
    winner = None
    if nextPlayer == 'X':
        moveNdx = 0
    else:
        moveNdx = 1
    while winner is None:
        player = playerList[moveNdx % 2]
        legalMovesMask = authority.LegalMovesMask(positionTensor, player)
        #print ("legalMovesMask = \n{}".format(legalMovesMask))
        chosenMoveTensor = neuralNetwork.ChooseAMove(
            positionTensor,
            player,
            authority,
            preApplySoftMax=True,
            softMaxTemperature=1.0
        )
        #print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
        positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
        #authority.Display(positionTensor)
        winner = authority.Winner(positionTensor)
        #print ("winner = {}".format(winner))
        moveNdx += 1
    if winner == nextPlayer:
        return 1.0
    elif winner == 'draw':
        return 0.5
    else:
        return 0

def main():
    print ("learnTicTacToe.py main()")

    authority = tictactoe.Authority()
    positionTensor = authority.InitialPosition()
    authority.Display(positionTensor)

    neuralNetwork = policy.NeuralNetwork(
        authority.PositionTensorShape(),
        '[(3, 32), (3, 32)]',
        authority.MoveTensorShape()
    )

    winner = None
    moveNdx = 0
    while winner is None:
        player = playerList[moveNdx % 2]
        legalMovesMask = authority.LegalMovesMask(positionTensor, player)
        print ("legalMovesMask = \n{}".format(legalMovesMask))
        chosenMoveTensor = neuralNetwork.ChooseAMove(
            positionTensor,
            player,
            authority,
            preApplySoftMax=True,
            softMaxTemperature=1.0
        )
        print ("chosenMoveTensor =\n{}".format(chosenMoveTensor))
        positionTensor, winner = authority.Move(positionTensor, player, chosenMoveTensor)
        authority.Display(positionTensor)
        winner = authority.Winner(positionTensor)
        print ("winner = {}".format(winner))
        moveNdx += 1

    rewardSum = 0
    numberOfTrials = 100
    for trial in range(numberOfTrials):
        initialPositionTensor = authority.InitialPosition()
        reward = SimulateGameAndGetReward(initialPositionTensor, 'X', authority, neuralNetwork)
        print ("reward = {}".format(reward))
        rewardSum += reward
    averageReward = rewardSum/numberOfTrials
    print ("averageReward = {}".format(averageReward))

if __name__ == '__main__':
    main()