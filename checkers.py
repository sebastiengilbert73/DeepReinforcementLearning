import torch
import gameAuthority

class Authority(gameAuthority.GameAuthority):
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor),
    #   Winner(self, positionTensor, lastPlayerWhoPlayed)
    #   LegalMovesMask(self, positionTensor, player)
    #   PositionTensorShape(self) done
    #   MoveTensorShape(self) done
    #   InitialPosition(self) done
    #   SwapPositions(self, positionTensor, player1, player2) done
    #   PlayersList(self) done
    #   MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString)
    #   Display(self, positionTensor) done

    def __init__(self):
        self.playersList= ['black', 'red']
        self.positionTensorShape = (4, 1, 8, 8)
        # 4 channels: Kings black, checkers black, checkers red, kings red
        self.moveTensorShape = (4, 1, 8, 8)
        # 4 channels: NW, NE, SE, SW
        # (H, W): origin of the moving piece
        self.pieceToPositionPlaneIndexDic = {'blackKing': 0, 'blackChecker': 1, 'redChecker': 2, 'redKing': 3}
        self.moveDirectionToMovePlaneIndexDic = {'NW': 0, 'NE': 1, 'SE': 2, 'SW': 3}

    def PositionTensorShape(self):
        return self.positionTensorShape

    def MoveTensorShape(self):
        return self.moveTensorShape

    def InitialPosition(self):
        initialPosition = torch.zeros(self.positionTensorShape)
        # Black checkers: channel = 1
        initialPosition[1, 0, 7, 0] = 1
        initialPosition[1, 0, 7, 2] = 1
        initialPosition[1, 0, 7, 4] = 1
        initialPosition[1, 0, 7, 6] = 1
        initialPosition[1, 0, 6, 1] = 1
        initialPosition[1, 0, 6, 3] = 1
        initialPosition[1, 0, 6, 5] = 1
        initialPosition[1, 0, 6, 7] = 1
        initialPosition[1, 0, 5, 0] = 1
        initialPosition[1, 0, 5, 2] = 1
        initialPosition[1, 0, 5, 4] = 1
        initialPosition[1, 0, 5, 6] = 1
        # Red checkers: channel = 2
        initialPosition[2, 0, 0, 1] = 1
        initialPosition[2, 0, 0, 3] = 1
        initialPosition[2, 0, 0, 5] = 1
        initialPosition[2, 0, 0, 7] = 1
        initialPosition[2, 0, 1, 0] = 1
        initialPosition[2, 0, 1, 2] = 1
        initialPosition[2, 0, 1, 4] = 1
        initialPosition[2, 0, 1, 6] = 1
        initialPosition[2, 0, 2, 1] = 1
        initialPosition[2, 0, 2, 3] = 1
        initialPosition[2, 0, 2, 5] = 1
        initialPosition[2, 0, 2, 7] = 1

        return initialPosition

    def PlayersList(self):
        return self.playersList

    def Display(self, positionTensor):
        channelNdxToSymbolDic = {0: ' bk ', 1: ' bc ', 2: ' rc ', 3: ' rk '}
        print ('   0   1   2   3   4   5   6   7')
        for row in range(8):
            print ('{} '.format(row), end='')
            for column in range(8):
                if positionTensor[0, 0, row, column] > 0:
                    print ('{}'.format(channelNdxToSymbolDic[0]), end='')
                elif positionTensor[1, 0, row, column] > 0:
                    print ('{}'.format(channelNdxToSymbolDic[1]), end='')
                elif positionTensor[2, 0, row, column] > 0:
                    print ('{}'.format(channelNdxToSymbolDic[2]), end='')
                elif positionTensor[3, 0, row, column] > 0:
                    print ('{}'.format(channelNdxToSymbolDic[3]), end='')
                else:
                    if (row + column) % 2 == 0: # White square
                        print ('    ', end='')
                    else: # Black square
                        print (' .  ', end='')
            print('\n')

    def SwapPositions(self, positionTensor, player1, player2):
        swappedPosition = torch.zeros(self.moveTensorShape)
        for row in range(8):
            for column in range(8):
                originalPiece = self.SquareOccupation(positionTensor, row, column)
                if originalPiece is 'blackKing':
                    swappedPosition[3, 0, 7 - row, 7 - column] = 1 # Put a red king in the mirror square
                elif originalPiece is 'blackChecker':
                    swappedPosition[2, 0, 7 - row, 7 - column] = 1 # Put a red checker in the mirror square
                elif originalPiece is 'redChecker':
                    swappedPosition[1, 0, 7 - row, 7 - column] = 1 # Put a black checker in the mirror square
                elif originalPiece is 'redKing':
                    swappedPosition[0, 0, 7 - row, 7 - column] = 1 # Put a black king in the mirror square
        return swappedPosition

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
            raise ValueError("Authority.Move(): moveTensor.shape ({}) is not {}".format(moveTensor.shape, self.moveTensorShape))
        # TODO: Check if the opponent can take a piece: if so, do nothing (the equivalent of giving back the turn for multiple jumps)

        newPositionTensor = currentPositionTensor.clone()
        nonZeroCoords = moveTensor.nonzero()
        print ("Move(): nonZeroCoords = {}".format(nonZeroCoords))
        print ("Move(): nonZeroCoords.shape = {}".format(nonZeroCoords.shape))
        if nonZeroCoords.shape[0] != 1:
            raise ValueError("checkers.py Move(): The number of non-zero values in the move tensor ({}) is not 1".format(nonZeroCoords.shape[0]))
        print ("Move(): nonZeroCoords[0] = {}".format(nonZeroCoords[0]))
        squareOccupation = self.SquareOccupation(currentPositionTensor, nonZeroCoords[0][2], nonZeroCoords[0][3])
        print ("squareOccupation = {}".format(squareOccupation))
        if squareOccupation is None:
            raise ValueError("checkers.py Move(): Attempt to move from an empty square ({}, {})".format(nonZeroCoords[0][2], nonZeroCoords[0][3]))
        if squareOccupation.startswith('red') and player is self.playersList[0]:
            raise ValueError("checkers.py Move(): Black player attempts to move a red piece in ({}, {})".format(nonZeroCoords[0][2], nonZeroCoords[0][3]))
        if squareOccupation.startswith('black') and player is self.playersList[1]:
            raise ValueError("checkers.py Move(): Red player attempts to move a black piece in ({}, {})".format(nonZeroCoords[0][2],
                                                                                                   nonZeroCoords[0][3]))
        if (squareOccupation is 'blackChecker') and \
            (nonZeroCoords[0][0] == 2 or nonZeroCoords[0][0] == 3):
            raise ValueError("checkers.py Move(): Attempt to move a black checker south in ({}, {})".format(nonZeroCoords[0][2],
                                                                                                   nonZeroCoords[0][3]))
        if (squareOccupation is 'redChecker') and \
            (nonZeroCoords[0][0] == 0 or nonZeroCoords[0][0] == 1):
            raise ValueError(
                "checkers.py Move(): Attempt to move a red checker north in ({}, {})".format(nonZeroCoords[0][2],
                                                                                               nonZeroCoords[0][3]))

        destinationSquare = [nonZeroCoords[0][2].item(), nonZeroCoords[0][3].item()]
        moveTypeIndex = nonZeroCoords[0][0]
        movingPieceIndex = self.pieceToPositionPlaneIndexDic[squareOccupation]
        if moveTypeIndex == 0: # NW
            destinationSquare[0] -= 1
            destinationSquare[1] -= 1
        elif moveTypeIndex == 1: # NE
            destinationSquare[0] -= 1
            destinationSquare[1] += 1
        elif moveTypeIndex == 2: # SE
            destinationSquare[0] += 1
            destinationSquare[1] += 1
        else: # 3 => SW
            destinationSquare[0] += 1
            destinationSquare[1] -= 1
        print ("Move(): destinationSquare = {}".format(destinationSquare))
        if destinationSquare[0] < 0 or destinationSquare[0] > 7 or destinationSquare[1] < 0 or destinationSquare[1] > 7:
            raise ValueError("checkers.py Move(): Attempt to move out of the checkerboard at ({}, {})".format(destinationSquare[0], destinationSquare[1]))

        destinationSquareOccupation = self.SquareOccupation(currentPositionTensor, destinationSquare[0], destinationSquare[1])
        if destinationSquareOccupation is None:
            newPositionTensor[movingPieceIndex, 0, nonZeroCoords[0][2], nonZeroCoords[0][3]] = 0
            newPositionTensor[movingPieceIndex, 0, destinationSquare[0], destinationSquare[1]] = 1
        else:
            if destinationSquareOccupation.startswith('black') and player is self.playersList[0]:
                raise ValueError("Move(): Black player attempts to move to a square occupied by a black piece in ({}, {})".format(destinationSquare[0], destinationSquare[1]))
            if destinationSquareOccupation.startswith('red') and player is self.playersList[1]:
                raise ValueError("Move(): Red player attempts to move to a square occupied by a red piece in ({}, {})".format(destinationSquare[0], destinationSquare[1]))

        return newPositionTensor, None

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        pass

    def LegalMovesMask(self, positionTensor, player):
        pass

    def MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString):
        pass

    def SquareOccupation(self, positionTensor, row, column):
        for (piece, planeNdx) in self.pieceToPositionPlaneIndexDic.items():
            if positionTensor[planeNdx, 0, row, column] > 0:
                return piece
        return None


def main():
    print ("checkers.py main()")
    authority = Authority()
    initialPosition = authority.InitialPosition()
    playersList = authority.PlayersList()

    authority.Display(initialPosition)
    moveTensor = torch.zeros(authority.MoveTensorShape())
    moveTensor[3, 0, 1, 2] = 1

    positionTensor, winner = authority.Move(initialPosition, playersList[1], moveTensor)
    authority.Display(positionTensor)

if __name__ == '__main__':
    main()