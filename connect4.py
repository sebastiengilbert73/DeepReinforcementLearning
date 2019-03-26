import torch
import policy

class Authority():
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor),
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self)
    #   MoveTensorShape(self)
    #   InitialPosition(self)
    #   SwapPositions(self, positionTensor, player1, player2)
    #   PlayersList(self)

    def __init__(self, numberOfRows=6, numberOfColumns=7):
        if numberOfColumns < 4 or numberOfRows < 4:
            raise ValueError("Authority.__init__(): The number of rows ({}) and the number of columns ({}) must be at least 4".format(numberOfRows, numberOfColumns))
        self.playersList= ['yellow', 'red']
        self.positionTensorShape = (2, 1, numberOfRows, numberOfColumns)
        self.moveTensorShape = (1, 1, 1, numberOfColumns)
        self.playerToPlaneIndexDic = {'yellow': 0, 'red': 1}
        self.numberOfRows = numberOfRows
        self.numberOfColumns = numberOfColumns

    def ThereIs4InARow(self, planeNdx, positionTensor):
        if positionTensor.shape != self.positionTensorShape: # (C, D, H, W)
            raise ValueError("Authority.ThereIs4InARow(): The shape of positionTensor ({}) is not {}".format(positionTensor.shape, self.positionTensorShape))
        # Horizontal lines
        for row in range(self.numberOfRows):
            for leftColumn in range(self.numberOfColumns - 4):
                thereIs4 = True
                for column in range(leftColumn, leftColumn + 4):
                    if positionTensor[planeNdx, 0, row, column] != 1:
                        thereIs4 = False
                if thereIs4:
                    return True

        # Vertical lines
        for column in range(self.numberOfColumns):
            for topRow in range(self.numberOfRows - 4):
                thereIs4 = True
                for row in range(topRow, topRow + 4):
                    if positionTensor[planeNdx, 0, row, column] != 1:
                        thereIs4 = False
                if thereIs4:
                    return True

        # Diagonal \
        for leftColumn in range(self.numberOfColumns - 3):
            for topRow in range(self.numberOfRows - 3):
                thereIs4 = True
                for index in range(4):
                    if positionTensor[planeNdx, 0, topRow + index, leftColumn + index] != 1:
                        thereIs4 = False
                if thereIs4:
                    return True

        # Diagonal /
        for leftColumn in range(self.numberOfColumns - 3):
            for bottomRow in range(3, self.numberOfRows):
                thereIs4 = True
                for index in range(4):
                    if positionTensor[planeNdx, 0, bottomRow - index, leftColumn + index] != 1:
                        thereIs4 = False
                if thereIs4:
                    return True

        # Otherwise
        return False

    