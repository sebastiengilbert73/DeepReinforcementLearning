import abc

class GameAuthority(abc.ABC):
    """
    Abstract class that holds the rules of the game
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def Move(self, currentPositionTensor, player, moveTensor):
        pass # return (positionTensor, winner)

    @abc.abstractmethod
    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        pass

    @abc.abstractmethod
    def LegalMovesMask(self, positionTensor, player):
        pass

    @abc.abstractmethod
    def PositionTensorShape(self):
        pass

    @abc.abstractmethod
    def MoveTensorShape(self):
        pass

    @abc.abstractmethod
    def InitialPosition(self):
        pass

    @abc.abstractmethod
    def SwapPositions(self, positionTensor, player1, player2):
        pass

    @abc.abstractmethod
    def PlayersList(self):
        pass

    @abc.abstractmethod
    def MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString):
        pass # return (positionTensor, winner)

    @abc.abstractmethod
    def Display(self, positionTensor):
        pass

    @abc.abstractmethod
    def RaiseAnErrorIfNoLegalMove(self):
        pass