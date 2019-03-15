import torch



class Authority():
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor), # return currentPositionTensor, winner
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self)
    #   MoveTensorShape(self)
    #   InitialPosition(self)
    #   SwapPositions(self, positionTensor, player1, player2)
    #   PlayersList(self)
    def __init__(self, target, maximumPlayedValue):
        self.target = target
        self.maximumPlayedValue = maximumPlayedValue
        self.moveTensorShape = (1, maximumPlayedValue, 1, 1)
        self.positionTensorShape = (1, target + 1, 1, 1)

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
            raise ValueError("Authority.Move(): moveTensor.shape ({}) != self.moveTensorShape ({})".format(moveTensor.shape, self.moveTensorShape))
        if currentPositionTensor.shape != self.positionTensorShape:
            raise ValueError("Authority.Move(): currentPositionTensor.shape ({}) != self.positionTensorShape ({})".format(currentPositionTensor.shape, self.positionTensorShape))
        if torch.nonzero(currentPositionTensor).size(0) != 1:
            raise ValueError("Authority.Move(): There should be a single non-zero value in currentPositionTensor. torch.nonzero(currentPositionTensor).size(0) = {}".forma(torch.nonzero(currentPositionTensor).size(0)))
        if torch.nonzero(moveTensor).size(0) != 1:
            raise ValueError("Authority.Move(): There should be a single non-zero value in moveTensor. torch.nonzero(moveTensor).size(0) = {}".format(torch.nonzero(moveTensor).size(0)))
        # Get the current sum
        currentSum = self.CurrentSum(currentPositionTensor)

        # Get the played value
        playedValue = self.PlayedValue(moveTensor)

        newSum = currentSum + playedValue
        if newSum > self.target:
            raise ValueError("Authority.Move(): The current sum is {} and the played value is {}. The sum can't exceed {}".format(currentSum, playedValue, self.target))

        newPositionTensor = torch.zeros(self.positionTensorShape)
        newPositionTensor[0, newSum, 0, 0] = 1
        return newPositionTensor, self.Winner(newPositionTensor, lastPlayerWhoPlayed=player)

    def CurrentSum(self, currentPositionTensor):
        currentSum = 0
        for index in range(self.positionTensorShape[1]):
            if currentPositionTensor[0, index, 0, 0] > 0:
                currentSum = index
        return currentSum

    def SetSum(self, sum):
        if sum < 0 or sum > self.target:
            raise ValueError("Authority.SetSum(): The sum ({}) is out of the range [0, {}]".format(sum, self.target))
        positionTensor = torch.zeros(self.positionTensorShape)
        positionTensor[0, sum, 0, 0] = 1
        return positionTensor

    def PlayedValue(self, moveTensor):
        playedValue = 0
        for index in range(self.moveTensorShape[1]):
            if moveTensor[0, index, 0, 0] > 0:
                playedValue = index + 1
        return playedValue

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        if positionTensor[0, self.target, 0, 0] > 0:
            return lastPlayerWhoPlayed
        return None

    def LegalMovesMask(self, positionTensor, player):
        currentSum = self.CurrentSum(positionTensor)
        legalMovesMask = torch.zeros(self.moveTensorShape).byte()
        for moveNdx in range(self.moveTensorShape[1]):
            if currentSum + moveNdx + 1 <= 100:
                legalMovesMask[0, moveNdx, 0, 0] = 1
        return legalMovesMask

    def PositionTensorShape(self):
        return self.positionTensorShape

    def MoveTensorShape(self):
        return self.moveTensorShape

    def InitialPosition(self):
        initialPositionTensor = torch.zeros(self.positionTensorShape)
        initialPositionTensor[0, 0, 0, 0] = 1
        return initialPositionTensor

    def SwapPositions(self, positionTensor, player1, player2):
        # Nothing to do
        return positionTensor

    def MoveWithInteger(self, currentPositionTensor, player, valueAsInteger):
        if valueAsInteger < 1 or valueAsInteger > self.maximumPlayedValue:
            raise ValueError("Authority.MoveWithInteger(): The value ({}) is out if the range [1, 10]".format(valueAsInteger))
        moveTensor = torch.zeros(self.moveTensorShape)
        moveTensor[0, valueAsInteger - 1, 0, 0] = 1
        return self.Move(currentPositionTensor, player, moveTensor)

    def PlayersList(self):
        playersList = []
        playersList.append('Player1')
        playersList.append('Player2')
        return playersList

    def MaximumPlayValue(self):
        return self.maximumPlayedValue

    def Target(self):
        return self.target

def main():
    print ("reachX.py main()")

    maximumPlayedValue = 10
    target = 12
    authority = Authority(target, maximumPlayedValue)

    positionTensorShape = authority.PositionTensorShape()
    moveTensorShape = authority.MoveTensorShape()
    playersList = authority.PlayersList()

    position = authority.InitialPosition()
    moveNdx = 0
    winner = None


    while winner is None:
        # Ask the player to choose a move
        player = playersList[moveNdx % 2]
        inputNbr = int(input ("Player {}: Sum = {}. Choose a number [1, {}]: ".format(player, authority.CurrentSum(position), maximumPlayedValue)))
        position, winner = authority.MoveWithInteger(position, player, inputNbr)

        moveNdx += 1
        # Swap the players
        position = authority.SwapPositions(position, playersList[0], playersList[1])


    print ("winner = {}".format(winner))


if __name__ == '__main__':
    main()