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

        moveVector = self.MoveVector(moveTypeIndex)
        destinationSquare[0] += moveVector[0]
        destinationSquare[1] += moveVector[1]

        print ("Move(): destinationSquare = {}".format(destinationSquare))
        if destinationSquare[0] < 0 or destinationSquare[0] > 7 or destinationSquare[1] < 0 or destinationSquare[1] > 7:
            raise ValueError("checkers.py Move(): Attempt to move out of the checkerboard at ({}, {})".format(destinationSquare[0], destinationSquare[1]))

        destinationSquareOccupation = self.SquareOccupation(currentPositionTensor, destinationSquare[0], destinationSquare[1])
        if destinationSquareOccupation is None:
            newPositionTensor[movingPieceIndex, 0, nonZeroCoords[0][2], nonZeroCoords[0][3]] = 0
            newPositionTensor[movingPieceIndex, 0, destinationSquare[0], destinationSquare[1]] = 1
        else: # The destination square is occupied: it should be a jump
            if destinationSquareOccupation.startswith('black') and player is self.playersList[0]:
                raise ValueError("checkers.py Move(): Black player attempts to move to a square occupied by a black piece in ({}, {})".format(destinationSquare[0], destinationSquare[1]))
            if destinationSquareOccupation.startswith('red') and player is self.playersList[1]:
                raise ValueError("checkers.py Move(): Red player attempts to move to a square occupied by a red piece in ({}, {})".format(destinationSquare[0], destinationSquare[1]))
            if destinationSquareOccupation.startswith('black') and player is self.playersList[1] or \
                    destinationSquareOccupation.startswith('red') and player is self.playersList[0]:
                jumpSquare = [destinationSquare[0] + moveVector[0], destinationSquare[1] + moveVector[1]] # jumpSquare is the square 2 diagonals away, where the piece will jump
                jumpSquareOccupation = self.SquareOccupation(currentPositionTensor, jumpSquare[0], jumpSquare[1])
                if jumpSquareOccupation is not None:
                    raise ValueError("checkers.py Move(): Trying to jump to an occupied square ({}, {})".format(jumpSquare[0], jumpSquare[1]))
                newPositionTensor[self.pieceToPositionPlaneIndexDic[destinationSquareOccupation], 0, destinationSquare[0], destinationSquare[1]] = 0
                newPositionTensor[movingPieceIndex, 0, nonZeroCoords[0][2], nonZeroCoords[0][3]] = 0
                newPositionTensor[movingPieceIndex, 0, jumpSquare[0], jumpSquare[1]] = 1

        # TODO: Check for coronation
        return newPositionTensor, None # TODO: check if there is a winner

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        pass

    def LegalMovesMask(self, positionTensor, player):
        if positionTensor.shape != self.positionTensorShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionTensor ({}) is not {}".format(
                positionTensor.shape, self.positionTensorShape))
        legalMovesMask = torch.zeros(self.moveTensorShape).byte()

        if player is self.playersList[0]:
            # Black kings
            blackKingsCoords = torch.nonzero(positionTensor[self.pieceToPositionPlaneIndexDic['blackKing']])
            for blackKingNdx in range(blackKingsCoords.shape[0]):
                blackKingCoords = [blackKingsCoords[blackKingNdx][1].item(), blackKingsCoords[blackKingNdx][2].item()] # Index 0 is the dummy 'depth' dimension
                # Move NW (0)
                moveVector = self.MoveVector(0)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[0, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[0, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                # Move NE (1)
                moveVector = self.MoveVector(1)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[1, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[1, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                # Move SE (2)
                moveVector = self.MoveVector(2)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[2, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[2, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                # Move SW (3)
                moveVector = self.MoveVector(3)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[3, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[3, 0, blackKingCoords[0], blackKingCoords[1]] = 1

            # Black checkers
            blackCheckersCoords = torch.nonzero( positionTensor[self.pieceToPositionPlaneIndexDic['blackChecker']])
            for blackCheckerNdx in range(blackCheckersCoords.shape[0]):
                blackCheckerCoords = [blackCheckersCoords[blackCheckerNdx][1].item(), blackCheckersCoords[blackCheckerNdx][2].item()] # Index 0 is the dummy 'depth' dimension
                #print ("blackCheckerCoords = {}".format(blackCheckerCoords))
                # Move NW (0)
                moveVector = self.MoveVector(0)
                firstCorner = [blackCheckerCoords[0] + moveVector[0], blackCheckerCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None: # Can move there
                        legalMovesMask[0, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                    elif firstCornerOccupation.startswith('red'): # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None: # Can jump there
                            legalMovesMask[0, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                # Move NE (1)
                moveVector = self.MoveVector(1)
                firstCorner = [blackCheckerCoords[0] + moveVector[0], blackCheckerCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[1, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                    elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[1, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1

        else: # Red
            # Red kings
            redKingsCoords = torch.nonzero(positionTensor[self.pieceToPositionPlaneIndexDic['redKing']])
            for redKingNdx in range(redKingsCoords.shape[0]):
                redKingCoords = [redKingsCoords[redKingNdx][1].item(), redKingsCoords[redKingNdx][2].item()] # Index 0 is the dummy 'depth' dimension
                # Move NW (0)
                moveVector = self.MoveVector(0)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[0, 0, redKingCoords[0], redKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[0, 0, redKingCoords[0], redKingCoords[1]] = 1
                # Move NE (1)
                moveVector = self.MoveVector(1)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[1, 0, redKingCoords[0], redKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[1, 0, redKingCoords[0], redKingCoords[1]] = 1
                # Move SE (2)
                moveVector = self.MoveVector(2)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[2, 0, redKingCoords[0], redKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[2, 0, redKingCoords[0], redKingCoords[1]] = 1
                # Move SW (3)
                moveVector = self.MoveVector(3)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[3, 0, redKingCoords[0], redKingCoords[1]] = 1
                    elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[3, 0, redKingCoords[0], redKingCoords[1]] = 1

            # Red checkers
            redCheckersCoords = torch.nonzero( positionTensor[self.pieceToPositionPlaneIndexDic['redChecker']])
            for redCheckerNdx in range(redCheckersCoords.shape[0]):
                redCheckerCoords = [redCheckersCoords[redCheckerNdx][1].item(), redCheckersCoords[redCheckerNdx][2].item()] # Index 0 is the dummy 'depth' dimension
                # Move SE (2)
                moveVector = self.MoveVector(2)
                firstCorner = [redCheckerCoords[0] + moveVector[0], redCheckerCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None: # Can move there
                        legalMovesMask[2, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1
                    elif firstCornerOccupation.startswith('black'): # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None: # Can jump there
                            legalMovesMask[2, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1
                # Move SW (3)
                moveVector = self.MoveVector(3)
                firstCorner = [redCheckerCoords[0] + moveVector[0], redCheckerCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[3, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1
                    elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[3, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1

        return legalMovesMask

    def MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString):
        pass

    def SquareOccupation(self, positionTensor, row, column):
        for (piece, planeNdx) in self.pieceToPositionPlaneIndexDic.items():
            if positionTensor[planeNdx, 0, row, column] > 0:
                return piece
        return None

    def MoveVector(self, moveTypeIndex):
        moveVector = [0, 0]
        if moveTypeIndex == 0:  # NW
            moveVector[0] -= 1
            moveVector[1] -= 1
        elif moveTypeIndex == 1:  # NE
            moveVector[0] -= 1
            moveVector[1] += 1
        elif moveTypeIndex == 2:  # SE
            moveVector[0] += 1
            moveVector[1] += 1
        else:  # 3 => SW
            moveVector[0] += 1
            moveVector[1] -= 1
        return moveVector

def main():
    print ("checkers.py main()")
    authority = Authority()
    playersList = authority.PlayersList()
    positionTensor = authority.InitialPosition()

    moveTensor = torch.zeros(authority.MoveTensorShape())
    moveTensor[1, 0, 5, 0] = 1
    positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)

    moveTensor = torch.zeros(authority.MoveTensorShape())
    moveTensor[3, 0, 2, 3] = 1
    positionTensor, winner = authority.Move(positionTensor, playersList[1], moveTensor)

    moveTensor = torch.zeros(authority.MoveTensorShape())
    moveTensor[1, 0, 4, 1] = 1
    positionTensor, winner = authority.Move(positionTensor, playersList[0], moveTensor)

    authority.Display(positionTensor)

    legalMovesMask = authority.LegalMovesMask(positionTensor, playersList[1])
    print ("legalMovesMask = \n{}".format(legalMovesMask))

if __name__ == '__main__':
    main()