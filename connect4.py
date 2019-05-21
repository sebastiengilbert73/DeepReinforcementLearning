import torch
import policy
import gameAuthority

class Authority(gameAuthority.GameAuthority):
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor),
    #   Winner(self, positionTensor, lastPlayerWhoPlayed) done
    #   LegalMovesMask(self, positionTensor, player) done
    #   PositionTensorShape(self) done
    #   MoveTensorShape(self) done
    #   InitialPosition(self) done
    #   SwapPositions(self, positionTensor, player1, player2) done
    #   PlayersList(self) done
    #   MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString) done
    #   Display(self, positionTensor)

    def __init__(self, numberOfRows=6, numberOfColumns=7):
        if numberOfColumns < 4 or numberOfRows < 4:
            raise ValueError("Authority.__init__(): The number of rows ({}) and the number of columns ({}) must be at least 4".format(numberOfRows, numberOfColumns))
        self.playersList= ['yellow', 'red']
        self.positionTensorShape = (2, 1, numberOfRows, numberOfColumns)
        self.moveTensorShape = (1, 1, 1, numberOfColumns)
        self.playerToPlaneIndexDic = {'yellow': 0, 'red': 1}
        self.numberOfRows = numberOfRows
        self.numberOfColumns = numberOfColumns

    def PlayersList(self):
        return self.playersList

    def ThereIs4InARow(self, planeNdx, positionTensor):
        if positionTensor.shape != self.positionTensorShape: # (C, D, H, W)
            raise ValueError("Authority.ThereIs4InARow(): The shape of positionTensor ({}) is not {}".format(positionTensor.shape, self.positionTensorShape))

        for row in range(self.numberOfRows):
            for column in range(self.numberOfColumns):
                # Left end of a horizontal line
                if column < self.numberOfColumns - 3:
                    thereIsAHorizontalLine = True
                    deltaColumn = 0
                    while deltaColumn < 4 and thereIsAHorizontalLine:
                        if positionTensor[planeNdx, 0, row , column + deltaColumn] != 1:
                            thereIsAHorizontalLine = False
                        deltaColumn += 1
                    if thereIsAHorizontalLine:
                        return True

                # Upper end of a vertical line
                if row < self.numberOfRows - 3:
                    thereIsAVerticalLine = True
                    deltaRow = 0
                    while deltaRow < 4 and thereIsAVerticalLine:
                        if positionTensor[planeNdx, 0, row + deltaRow, column] != 1:
                            thereIsAVerticalLine = False
                        deltaRow += 1
                    if thereIsAVerticalLine:
                        return True

                # North-West end of a \
                if row < self.numberOfRows - 3 and column < self.numberOfColumns - 3:
                    thereIsABackSlash = True
                    deltaRowColumn = 0
                    while deltaRowColumn < 4 and thereIsABackSlash:
                        if positionTensor[planeNdx, 0, row + deltaRowColumn, column + deltaRowColumn] != 1:
                            thereIsABackSlash = False
                        deltaRowColumn += 1
                    if thereIsABackSlash:
                        return True

                # South-West end of a /
                if row < self.numberOfRows - 3 and column >= 3:
                    thereIsASlash = True
                    deltaRowColumn = 0
                    while deltaRowColumn < 4 and thereIsASlash:
                        if positionTensor[planeNdx, 0, row + deltaRowColumn, column - deltaRowColumn] != 1:
                            thereIsASlash = False
                        deltaRowColumn += 1
                    if thereIsASlash:
                        return True
        # Otherwise
        return  False

        """# Horizontal lines
        for row in range(self.numberOfRows):
            for leftColumn in range(self.numberOfColumns - 3):
                thereIs4 = True
                for column in range(leftColumn, leftColumn + 4):
                    if positionTensor[planeNdx, 0, row, column] != 1:
                        thereIs4 = False
                if thereIs4:
                    return True

        # Vertical lines
        for column in range(self.numberOfColumns):
            for topRow in range(self.numberOfRows - 3):
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
        """


    def MoveWithColumn(self, currentPositionTensor, player, dropColumn):
        #print ("MoveWithColumn(): currentPositionTensor =\n{}".format(currentPositionTensor))
        if currentPositionTensor.shape != self.positionTensorShape: # (C, D, H, W)
            raise ValueError("Authority.MoveWithColumn(): The shape of currentPositionTensor {} is not {}".format(currentPositionTensor.shape, self.positionTensorShape))
        if dropColumn >= self.numberOfColumns:
            raise ValueError("Authority.MoveWithColumn(): dropColumn ({}) is >= self.numberOfColumns ({})".format(dropColumn, self.numberOfColumns))
        #print ("Authority.MoveWithColumn(): currentPositionTensor.shape = {}".format(currentPositionTensor.shape))
        newPositionTensor = currentPositionTensor.clone()
        topAvailableRow = self.TopAvailableRow(currentPositionTensor, dropColumn)
        if topAvailableRow == None:
            raise ValueError(
                "Authority.MoveWithColumn(): Attempt to drop in column {}, while it is already filled".format(
                    dropColumn))
        newPositionTensor[self.playerToPlaneIndexDic[player], 0, topAvailableRow, dropColumn] = 1.0
        #print ("Authority.MoveWithColumn(): newPositionTensor.shape = {}".format(newPositionTensor.shape))
        winner = self.Winner(newPositionTensor, player)
        return newPositionTensor, winner

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
            raise ValueError("Authority.Move(): moveTensor.shape ({}) is not {}".format(moveTensor.shape, self.moveTensorShape))
        numberOfOnes = 0
        dropColumn = None

        for column in range(self.numberOfColumns):
            if moveTensor[0, 0, 0, column] == 1:
                numberOfOnes += 1
                dropColumn = column
        if numberOfOnes != 1:
            raise ValueError("Authority.Move(): The number of ones in moveTensor ({}) is not one".format(numberOfOnes))
        return self.MoveWithColumn(currentPositionTensor, player, dropColumn)

    def MoveInPlace(self, currentPositionTensor, player, moveTensor):
        dropColumn = self.DropColumn(moveTensor)
        topAvailableRow = self.TopAvailableRow(currentPositionTensor, dropColumn)
        if topAvailableRow == None:
            raise ValueError(
                "Authority.MoveInPlace(): Attempt to drop in column {}, while it is already filled".format(
                    dropColumn))
        currentPositionTensor[self.playerToPlaneIndexDic[player], 0, topAvailableRow, dropColumn] = 1.0
        winner = self.Winner(currentPositionTensor, player)
        return winner

    def DropColumn(self, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
            raise ValueError("Authority.DropColumn(): moveTensor.shape ({}) is not {}".format(moveTensor.shape,                                                                                             self.moveTensorShape))
        numberOfOnes = 0
        dropColumn = None
        for column in range(self.numberOfColumns):
            if moveTensor[0, 0, 0, column] == 1:
                numberOfOnes += 1
                dropColumn = column
        if numberOfOnes != 1:
            raise ValueError("Authority.DropColumn(): The number of ones in moveTensor ({}) is not one".format(numberOfOnes))
        return dropColumn

    def MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString):
        dropColumn = int(dropCoordinatesAsString)
        if dropColumn < 0 or dropColumn >= self.numberOfColumns:
            raise ValueError("Authority.MoveWithString(): The drop column ({}) is not in [0, {}]".format(dropColumn, self.numberOfColumns - 1))
        return self.MoveWithColumn(currentPositionTensor, player, dropColumn)

    def TopAvailableRow(self, positionTensor, dropColumn):
        # Must return None if the column is already filled
        # Check the bottom row
        if positionTensor[0, 0, self.numberOfRows - 1, dropColumn] == 0 and \
            positionTensor[1, 0, self.numberOfRows - 1, dropColumn] == 0:
            return self.numberOfRows - 1

        highestOneRow = self.numberOfRows - 1
        for row in range(self.numberOfRows - 2, -1, -1): # Cound backward: 4, 3, 2, 1, 0
            if positionTensor[0, 0, row, dropColumn] > 0 or \
                positionTensor[1, 0, row, dropColumn] > 0:
                highestOneRow = row
        if highestOneRow == 0: # The column is already filled
            return None
        else:
            return highestOneRow - 1


    def InitialPosition(self):
        initialPosition = torch.zeros(self.positionTensorShape)
        return initialPosition

    def MoveTensorShape(self):
        return self.moveTensorShape

    def PositionTensorShape(self):
        return self.positionTensorShape

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        lastPlayerPlane = self.playerToPlaneIndexDic[lastPlayerWhoPlayed]
        if self.ThereIs4InARow(lastPlayerPlane, positionTensor):
            return lastPlayerWhoPlayed
        else:
            if torch.nonzero(positionTensor).size(0) == self.numberOfRows * self.numberOfColumns: # All spots are occupied
                return 'draw'
            else:
                return None

    def LegalMovesMask(self, positionTensor, player):
        if positionTensor.shape != self.positionTensorShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionTensor ({}) is not {}".format(
                positionTensor.shape, self.positionTensorShape))
        legalMovesMask = torch.zeros(self.moveTensorShape).byte() + 1  # Initialized with ones, i.e legal moves
        for column in range(self.numberOfColumns):
            if positionTensor[0, 0, 0, column] != 0 or positionTensor[1, 0, 0, column] != 0:
                legalMovesMask[0, 0, 0, column] = 0
        return legalMovesMask

    def SwapPositions(self, positionTensor, player1, player2):
        player1PlaneNdx = self.playerToPlaneIndexDic[player1]
        player2PlaneNdx = self.playerToPlaneIndexDic[player2]
        """swappedPosition = positionTensor.clone()
        swappedPosition[player1PlaneNdx] = positionTensor[player2PlaneNdx]
        swappedPosition[player2PlaneNdx] = positionTensor[player1PlaneNdx]
        """
        swappedPosition = torch.index_select(positionTensor, 0, torch.LongTensor([player2PlaneNdx, player1PlaneNdx]))
        """swappedPosition = torch.zeros(self.positionTensorShape)
        nonZeroCoordsTensor = torch.nonzero(positionTensor)
        for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
            nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
            if nonZeroCoords[0] == player1PlaneNdx:
                swappedPosition[player2PlaneNdx, nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
            else:
                swappedPosition[player1PlaneNdx, nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
        """
        return swappedPosition

    def Display(self, positionTensor):
        print ("Authority.Display(): positionTensor =\n{}".format(positionTensor))
        planeNdxToSymbolDic = {0: 'y', 1: 'r'}
        for row in range(self.numberOfRows):
            for column in range(self.numberOfColumns):
                if positionTensor[0, 0, row, column] > 0:
                    print ('{} '.format(planeNdxToSymbolDic[0]), end='')
                elif positionTensor[1, 0, row, column] > 0:
                    print ('{} '.format(planeNdxToSymbolDic[1]), end='')
                else:
                    print ('. ', end='')
            print('\n')



def main():
    print ("connect4.py main()")
    authority = Authority()
    playersList = authority.PlayersList()
    position = authority.InitialPosition()
    #position, winner = authority.MoveWithColumn(position, playersList[0], 2)
    #position, winner = authority.MoveWithColumn(position, playersList[1], 2)
    #position, winner = authority.MoveWithColumn(position, playersList[0], 2)
    #position, winner = authority.MoveWithColumn(position, playersList[1], 2)
    position, winner = authority.MoveWithColumn(position, playersList[0], 6)
    position, winner = authority.MoveWithColumn(position, playersList[1], 5)
    position, winner = authority.MoveWithColumn(position, playersList[0], 5)
    position, winner = authority.MoveWithColumn(position, playersList[1], 4)
    position, winner = authority.MoveWithColumn(position, playersList[1], 4)
    position, winner = authority.MoveWithColumn(position, playersList[0], 4)
    position, winner = authority.MoveWithColumn(position, playersList[0], 3)
    position, winner = authority.MoveWithColumn(position, playersList[1], 3)
    position, winner = authority.MoveWithColumn(position, playersList[1], 3)
    #position, winner = authority.MoveWithColumn(position, playersList[0], 3)

    moveTensor = torch.zeros(authority.MoveTensorShape())
    moveTensor[0, 0, 0, 3] = 1
    winner = authority.MoveInPlace(position, 'red', moveTensor)
    print ("position =\n{}".format(position))
    authority.Display(position)
    print ("winner = {}".format(winner))

if __name__ == '__main__':
    main()


