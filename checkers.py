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
        pass

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
    initialPosition[1, 0, 5, 0] = 0
    initialPosition[0, 0, 5, 0] = 1

    initialPosition[1, 0, 5, 4] = 0

    initialPosition[2, 0, 0, 7] = 0
    initialPosition[3, 0, 0, 7] = 1
    authority.Display(initialPosition)

    print ("authority.SquareOccupation(initialPosition, 5, 0) = {}".format(authority.SquareOccupation(initialPosition, 5, 0)))
    playersList = authority.PlayersList()
    swappedPosition = authority.SwapPositions(initialPosition, playersList[0], playersList[1])
    authority.Display(swappedPosition)

if __name__ == '__main__':
    main()