import torch

player1Name = 'Player1'
player2Name = 'Player2'

positionTensorShape = (1, 101, 1, 1)
moveTensorShape = (1, 10, 1, 1)

class Authority():
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor), # return currentPositionTensor, winner
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self)
    #   MoveTensorShape(self)
    #   InitialPosition(self)
    #   SwapPositions(self, positionTensor, player1, player2)
    def __init__(self):
        pass

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != moveTensorShape:
            raise ValueError("Authority.Move(): moveTensor.shape ({}) != moveTensorShape ({})".format(moveTensor.shape, moveTensorShape))
        if currentPositionTensor.shape != positionTensorShape:
            raise ValueError("Authority.Move(): currentPositionTensor.shape ({}) != positionTensorShape ({})".format(currentPositionTensor.shape, positionTensorShape))
        if torch.nonzero(currentPositionTensor).size(0) != 1:
            raise ValueError("Authority.Move(): There should be a single non-zero value in currentPositionTensor. torch.nonzero(currentPositionTensor).size(0) = {}".forma(torch.nonzero(currentPositionTensor).size(0)))
        if torch.nonzero(moveTensor).size(0) != 1:
            raise ValueError("Authority.Move(): There should be a single non-zero value in moveTensor. torch.nonzero(moveTensor).size(0) = {}".format(torch.nonzero(moveTensor).size(0)))
        # Get the current sum
        currentSum = self.CurrentSum(currentPositionTensor)

        # Get the played value
        playedValue = self.PlayedValue(moveTensor)

        newSum = currentSum + playedValue
        if newSum > 100:
            raise ValueError("Authority.Move(): The current sum is {} and the played value is {}. The sum can't exceed 100".format(currentSum, playedValue))

        newPositionTensor = torch.zeros(positionTensorShape)
        newPositionTensor[0, newSum, 0, 0] = 1
        return newPositionTensor, self.Winner(newPositionTensor, lastPlayerWhoPlayed=player)

    def CurrentSum(self, currentPositionTensor):
        currentSum = 0
        for index in range(positionTensorShape[1]):
            if currentPositionTensor[0, index, 0, 0] > 0:
                currentSum = index
        return currentSum

    def SetSum(self, sum):
        if sum < 0 or sum > 100:
            raise ValueError("Authority.SetSum(): The sum ({}) is out of the range [0, 100]".format(sum))
        positionTensor = torch.zeros(positionTensorShape)
        positionTensor[0, sum, 0, 0] = 1
        return positionTensor

    def PlayedValue(self, moveTensor):
        playedValue = 0
        for index in range(moveTensorShape[1]):
            if moveTensor[0, index, 0, 0] > 0:
                playedValue = index + 1
        return playedValue

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        if positionTensor[0, 100, 0, 0] > 0:
            return lastPlayerWhoPlayed
        return None

    def LegalMovesMask(self, positionTensor, player):
        currentSum = self.CurrentSum(positionTensor)
        legalMovesMask = torch.zeros(moveTensorShape).byte()
        for moveNdx in range(moveTensorShape[1]):
            if currentSum + moveNdx + 1 <= 100:
                legalMovesMask[0, moveNdx, 0, 0] = 1
        return legalMovesMask

    def PositionTensorShape(self):
        return positionTensorShape

    def MoveTensorShape(self):
        return moveTensorShape

    def InitialPosition(self):
        initialPositionTensor = torch.zeros(positionTensorShape)
        initialPositionTensor[0, 0, 0, 0] = 1
        return initialPositionTensor

    def SwapPositions(self, positionTensor, player1, player2):
        # Nothing to do
        return positionTensor

    def MoveWithInteger(self, currentPositionTensor, player, valueAsInteger):
        if valueAsInteger < 1 or valueAsInteger > 10:
            raise ValueError("Authority.MoveWithInteger(): The value ({}) is out if the range [1, 10]".format(valueAsInteger))
        moveTensor = torch.zeros(moveTensorShape)
        moveTensor[0, valueAsInteger - 1, 0, 0] = 1
        return self.Move(currentPositionTensor, player, moveTensor)

def main():
    print ("reach100.py main()")


if __name__ == '__main__':
    main()