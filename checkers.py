import torch
import gameAuthority

class Authority(gameAuthority.GameAuthority):
    # Must implement:
    #   Move(self, currentPositionTensor, player, moveTensor), done
    #   Winner(self, positionTensor, lastPlayerWhoPlayed) done
    #   LegalMovesMask(self, positionTensor, player) done
    #   PositionTensorShape(self) done
    #   MoveTensorShape(self) done
    #   InitialPosition(self) done
    #   SwapPositions(self, positionTensor, player1, player2) done
    #   PlayersList(self) done
    #   MoveWithString(self, currentPositionTensor, player, dropCoordinatesAsString) done
    #   Display(self, positionTensor) done

    def __init__(self):
        self.playersList= ['black', 'red']
        self.positionTensorShape = (6, 1, 8, 8)
        # 4 channels: Kings black, checkers black, checkers red, kings red, black last move's jump (if any), red last move's jump (if any)
        self.moveTensorShape = (4, 1, 8, 8)
        # 4 channels: NW, NE, SE, SW
        # (H, W): origin of the moving piece
        self.pieceToPositionPlaneIndexDic = {'blackKing': 0, 'blackChecker': 1, 'redChecker': 2, 'redKing': 3, 'lastBlackJump': 4, 'lastRedJump': 5}
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
        swappedPosition = torch.zeros(self.positionTensorShape)
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
                if positionTensor[4, 0, row, column] == 1:
                    swappedPosition[5, 0, 7 - row, 7 - column] = 1
                if positionTensor[5, 0, row, column] == 1:
                    swappedPosition[4, 0, 7 - row, 7 - column] = 1

        return swappedPosition

    def Move(self, currentPositionTensor, player, moveTensor):
        if moveTensor.shape != self.moveTensorShape:
            raise ValueError("Authority.Move(): moveTensor.shape ({}) is not {}".format(moveTensor.shape, self.moveTensorShape))
        # Check if the opponent can take a piece: if so, do nothing (the equivalent of giving back the turn for multiple jumps)
        if player is self.playersList[0] and torch.nonzero(currentPositionTensor[self.pieceToPositionPlaneIndexDic['lastRedJump']]).shape[0] > 0:
            redJumperLocation = self.LastJumperLocation(currentPositionTensor, self.pieceToPositionPlaneIndexDic['lastRedJump'])
            redPossibleCaptures = self.PossibleCaptures(currentPositionTensor, self.playersList[1])
            if torch.max(redPossibleCaptures[:, 0, redJumperLocation[0], redJumperLocation[1]]).item() > 0:
                return currentPositionTensor, None
        elif player is self.playersList[1] and torch.nonzero(currentPositionTensor[self.pieceToPositionPlaneIndexDic['lastBlackJump']]).shape[0] > 0:
            blackJumperLocation = self.LastJumperLocation(currentPositionTensor, self.pieceToPositionPlaneIndexDic['lastBlackJump'])
            blackPossibleCaptures = self.PossibleCaptures(currentPositionTensor, self.playersList[0])
            if torch.max(blackPossibleCaptures[:, 0, blackJumperLocation[0], blackJumperLocation[1]]).item() > 0:
                return currentPositionTensor, None



        newPositionTensor = currentPositionTensor.clone()
        newPositionTensor[self.pieceToPositionPlaneIndexDic['lastBlackJump']] = 0
        newPositionTensor[self.pieceToPositionPlaneIndexDic['lastRedJump']] = 0
        nonZeroCoords = moveTensor.nonzero()
        #print ("Move(): nonZeroCoords = {}".format(nonZeroCoords))
        #print ("Move(): nonZeroCoords.shape = {}".format(nonZeroCoords.shape))
        if nonZeroCoords.shape[0] != 1:
            raise ValueError("checkers.py Move(): The number of non-zero values in the move tensor ({}) is not 1".format(nonZeroCoords.shape[0]))
        #print ("Move(): nonZeroCoords[0] = {}".format(nonZeroCoords[0]))


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

        # Check if the move is legal
        legalMovesMask = self.LegalMovesMask(currentPositionTensor, player)
        if legalMovesMask[nonZeroCoords[0][0], nonZeroCoords[0][1], nonZeroCoords[0][2], nonZeroCoords[0][3]].item() == 0:
            raise ValueError("checkers.py Move(): Attempt to do an illegal move: {}".format(nonZeroCoords[0]))

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
            """if player is self.playersList[0]:
                newPositionTensor[self.pieceToPositionPlaneIndexDic['lastBlackJump']] = 0
            else:
                newPositionTensor[self.pieceToPositionPlaneIndexDic['lastRedJump']] = 0"""
        else: # The destination square is occupied: it should be a jump
            if destinationSquareOccupation.startswith('black') and player is self.playersList[0]:
                raise ValueError("checkers.py Move(): Black player attempts to move to a square occupied by a black piece in ({}, {})".format(destinationSquare[0], destinationSquare[1]))
            if destinationSquareOccupation.startswith('red') and player is self.playersList[1]:
                raise ValueError("checkers.py Move(): Red player attempts to move to a square occupied by a red piece in ({}, {})".format(destinationSquare[0], destinationSquare[1]))
            if destinationSquareOccupation.startswith('black') and player is self.playersList[1] or \
                    destinationSquareOccupation.startswith('red') and player is self.playersList[0]:
                jumpSquare = [destinationSquare[0] + moveVector[0], destinationSquare[1] + moveVector[1]] # jumpSquare is the square 2 diagonals away, where the piece will jump
                if not self.IsInCheckerboard(jumpSquare):
                    raise ValueError("checkers.py Move(): Attempt to jump out of the checkerboard {}".format(jumpSquare))
                jumpSquareOccupation = self.SquareOccupation(currentPositionTensor, jumpSquare[0], jumpSquare[1])
                if jumpSquareOccupation is not None:
                    raise ValueError("checkers.py Move(): Trying to jump to an occupied square ({}, {})".format(jumpSquare[0], jumpSquare[1]))
                newPositionTensor[self.pieceToPositionPlaneIndexDic[destinationSquareOccupation], 0, destinationSquare[0], destinationSquare[1]] = 0
                newPositionTensor[movingPieceIndex, 0, nonZeroCoords[0][2], nonZeroCoords[0][3]] = 0
                newPositionTensor[movingPieceIndex, 0, jumpSquare[0], jumpSquare[1]] = 1
                if player is self.playersList[0]:
                    newPositionTensor[self.pieceToPositionPlaneIndexDic['lastBlackJump'], 0, jumpSquare[0], jumpSquare[1]] = 1
                else:
                    newPositionTensor[self.pieceToPositionPlaneIndexDic['lastRedJump'], 0, jumpSquare[0], jumpSquare[1]] = 1
                destinationSquare = jumpSquare

        if squareOccupation is 'blackChecker' and destinationSquare[0] == 0:
            newPositionTensor[self.pieceToPositionPlaneIndexDic['blackChecker'], 0, destinationSquare[0], destinationSquare[1]] = 0
            newPositionTensor[self.pieceToPositionPlaneIndexDic['blackKing'], 0, destinationSquare[0], destinationSquare[1]] = 1
        if squareOccupation is 'redChecker' and destinationSquare[0] == 7:
            newPositionTensor[self.pieceToPositionPlaneIndexDic['redChecker'], 0, destinationSquare[0], destinationSquare[1]] = 0
            newPositionTensor[self.pieceToPositionPlaneIndexDic['redKing'], 0, destinationSquare[0], destinationSquare[1]] = 1
        print ("Move(): newPositionTensor.shape = {}".format(newPositionTensor.shape))
        return newPositionTensor, self.Winner(newPositionTensor, player)

    def Winner(self, positionTensor, lastPlayerWhoPlayed):
        if lastPlayerWhoPlayed is self.playersList[0]:
            if torch.nonzero(positionTensor[self.pieceToPositionPlaneIndexDic['lastBlackJump']]).shape[0] > 0: # Black just took a piece
                jumperLocation = self.LastJumperLocation(positionTensor, self.pieceToPositionPlaneIndexDic['lastBlackJump'])
                blackPossibleCaptures = self.PossibleCaptures(positionTensor, self.playersList[0])
                if torch.max(blackPossibleCaptures[:, 0, jumperLocation[0], jumperLocation[1]]).item() > 0:
                    return None
            redLegalMoves = self.LegalMovesMask(positionTensor, self.playersList[1])
            if torch.nonzero(redLegalMoves).shape[0] == 0:
                return self.playersList[0]
            else:
                return None
        else: # Red played last
            if torch.nonzero(positionTensor[self.pieceToPositionPlaneIndexDic['lastRedJump']]).shape[0] > 0: # Red just took a piece
                jumperLocation = self.LastJumperLocation(positionTensor, self.pieceToPositionPlaneIndexDic['lastRedJump'])
                redPossibleCaptures = self.PossibleCaptures(positionTensor, self.playersList[1])
                if torch.max(redPossibleCaptures[:, 0, jumperLocation[0], jumperLocation[1]]).item() > 0:
                    return None
            blackLegalMoves = self.LegalMovesMask(positionTensor, self.playersList[0])
            if torch.nonzero(blackLegalMoves).shape[0] == 0:
                return self.playersList[1]
            else:
                return None

    def LastJumperLocation(self, positionTensor, planeIndex):
        lastJumperLocations = []
        for row in range(8):
            for column in range(8):
                if positionTensor[planeIndex, 0, row, column] > 0:
                    lastJumperLocations.append([row, column])
        if len(lastJumperLocations) == 0:
            return None
        if len(lastJumperLocations) >= 2:
            raise ValueError("LastJumperLocation(): len(lastJumperLocations) ({}) >= 2".format(len(lastJumperLocations)))
        return lastJumperLocations[0] # The one and only 1 value

    def LegalMovesMask(self, positionTensor, player):
        if positionTensor.shape != self.positionTensorShape:
            raise ValueError("Authority.LegalMovesMask(): The shape of positionTensor ({}) is not {}".format(
                positionTensor.shape, self.positionTensorShape))
        legalMovesMask = self.PossibleCaptures(positionTensor, player)
        # Check if there is at lease one possible capture
        if torch.nonzero(legalMovesMask).shape[0] > 0:
            return legalMovesMask

        # No possible capture: Consider the non-capturing moves
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
                    """elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[0, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """
                # Move NE (1)
                moveVector = self.MoveVector(1)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[1, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[1, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """
                # Move SE (2)
                moveVector = self.MoveVector(2)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[2, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[2, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """
                # Move SW (3)
                moveVector = self.MoveVector(3)
                firstCorner = [blackKingCoords[0] + moveVector[0], blackKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[3, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[3, 0, blackKingCoords[0], blackKingCoords[1]] = 1
                    """

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
                    """elif firstCornerOccupation.startswith('red'): # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None: # Can jump there
                            legalMovesMask[0, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                    """
                # Move NE (1)
                moveVector = self.MoveVector(1)
                firstCorner = [blackCheckerCoords[0] + moveVector[0], blackCheckerCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[1, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('red'):  # first corner is occupied by a red piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[1, 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                    """

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
                    """elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[0, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """
                # Move NE (1)
                moveVector = self.MoveVector(1)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[1, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[1, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """
                # Move SE (2)
                moveVector = self.MoveVector(2)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[2, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[2, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """
                # Move SW (3)
                moveVector = self.MoveVector(3)
                firstCorner = [redKingCoords[0] + moveVector[0], redKingCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0],
                                                                  firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[3, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0],
                                                                       secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[3, 0, redKingCoords[0], redKingCoords[1]] = 1
                    """

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
                    """elif firstCornerOccupation.startswith('black'): # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None: # Can jump there
                            legalMovesMask[2, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1
                    """
                # Move SW (3)
                moveVector = self.MoveVector(3)
                firstCorner = [redCheckerCoords[0] + moveVector[0], redCheckerCoords[1] + moveVector[1]]
                if firstCorner[0] >= 0 and firstCorner[0] < 8 and firstCorner[1] >= 0 and firstCorner[1] < 8:
                    firstCornerOccupation = self.SquareOccupation(positionTensor, firstCorner[0], firstCorner[1])
                    if firstCornerOccupation is None:  # Can move there
                        legalMovesMask[3, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1
                    """elif firstCornerOccupation.startswith('black'):  # first corner is occupied by a black piece
                        secondCorner = [firstCorner[0] + moveVector[0], firstCorner[1] + moveVector[1]]
                        secondCornerOccupation = self.SquareOccupation(positionTensor, secondCorner[0], secondCorner[1])
                        if secondCornerOccupation is None:  # Can jump there
                            legalMovesMask[3, 0, redCheckerCoords[0], redCheckerCoords[1]] = 1
                    """

        return legalMovesMask

    def MoveWithString(self, currentPositionTensor, player, moveAsString):
        # Expected string: '52-43': Move the piece in (5, 2) to (4, 3)
        if len(moveAsString) != 5:
            raise ValueError("MoveWithString(): The length of the string {} is not 5".format(moveAsString))
        startArrivalSquares = moveAsString.split('-')
        if len(startArrivalSquares) != 2:
            raise ValueError("MoveWithString(): The split on '-' of {} doesn't give two parts.".format(moveAsString))
        startSquare = [int(startArrivalSquares[0][0]), int(startArrivalSquares[0][1])]
        arrivalSquare = [int(startArrivalSquares[1][0]), int(startArrivalSquares[1][1])]
        moveTypeIndex = -1
        if arrivalSquare[0] - startSquare[0] < 0 and arrivalSquare[1] - startSquare[1] < 0: # NW
            moveTypeIndex = 0
        elif arrivalSquare[0] - startSquare[0] < 0 and arrivalSquare[1] - startSquare[1] > 0: # NE
            moveTypeIndex = 1
        elif arrivalSquare[0] - startSquare[0] > 0 and arrivalSquare[1] - startSquare[1] > 0: # SE
            moveTypeIndex = 2
        elif arrivalSquare[0] - startSquare[0] > 0 and arrivalSquare[1] - startSquare[1] < 0: # SW
            moveTypeIndex = 3
        else:
            raise ValueError("MoveWithString(): One of the coordinates is identical: {}".format(moveAsString))
        moveTensor = torch.zeros(self.moveTensorShape)
        moveTensor[moveTypeIndex, 0, startSquare[0], startSquare[1]] = 1
        return self.Move(currentPositionTensor, player, moveTensor)

    def SquareOccupation(self, positionTensor, row, column):
        if row < 0 or row > 7 or column < 0 or column > 7:
            return None
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

    def PossibleCaptures(self, positionTensor, player):
        possibleCaptures = torch.zeros(self.moveTensorShape).byte()
        if player is self.playersList[0]: # Black
            # Black kings
            blackKingsCoords = torch.nonzero(positionTensor[self.pieceToPositionPlaneIndexDic['blackKing']])
            for blackKingNdx in range(blackKingsCoords.shape[0]):
                blackKingCoords = [blackKingsCoords[blackKingNdx][1].item(), blackKingsCoords[blackKingNdx][2].item()]
                # NW
                firstDiagonal = [blackKingCoords[0] - 1, blackKingCoords[1] - 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0], firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('red'):
                    secondDiagonal = [blackKingCoords[0] - 2, blackKingCoords[1] - 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0], secondDiagonal[1])
                    if self.IsInCheckerboard(secondDiagonal) and secondDiagonalOccupation is None: # Possible capture
                        possibleCaptures[self.moveDirectionToMovePlaneIndexDic['NW'], 0, blackKingCoords[0], blackKingCoords[1]] = 1
                # NE
                firstDiagonal = [blackKingCoords[0] - 1, blackKingCoords[1] + 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0], firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('red'):
                    secondDiagonal = [blackKingCoords[0] - 2, blackKingCoords[1] + 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['NE'], 0, blackKingCoords[0], blackKingCoords[1]] = 1
                # SE
                firstDiagonal = [blackKingCoords[0] + 1, blackKingCoords[1] + 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('red'):
                    secondDiagonal = [blackKingCoords[0] + 2, blackKingCoords[1] + 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['SE'], 0, blackKingCoords[0], blackKingCoords[
                                1]] = 1
                # SW
                firstDiagonal = [blackKingCoords[0] + 1, blackKingCoords[1] - 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('red'):
                    secondDiagonal = [blackKingCoords[0] + 2, blackKingCoords[1] - 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['SW'], 0, blackKingCoords[0], blackKingCoords[
                                1]] = 1
            # Black checkers
            blackCheckersCoords = torch.nonzero(
                positionTensor[self.pieceToPositionPlaneIndexDic['blackChecker']])
            for blackCheckerNdx in range(blackCheckersCoords.shape[0]):
                blackCheckerCoords = [blackCheckersCoords[blackCheckerNdx][1].item(),
                                      blackCheckersCoords[blackCheckerNdx][2].item()]
                # NW
                firstDiagonal = [blackCheckerCoords[0] - 1, blackCheckerCoords[1] - 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0], firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('red'):
                    secondDiagonal = [blackCheckerCoords[0] - 2, blackCheckerCoords[1] - 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['NW'], 0, blackCheckerCoords[0], blackCheckerCoords[1]] = 1
                # NE
                firstDiagonal = [blackCheckerCoords[0] - 1, blackCheckerCoords[1] + 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('red'):
                    secondDiagonal = [blackCheckerCoords[0] - 2, blackCheckerCoords[1] + 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['NE'], 0, blackCheckerCoords[0], blackCheckerCoords[
                                1]] = 1

        else: # Red
            # Red kings
            redKingsCoords = torch.nonzero(positionTensor[self.pieceToPositionPlaneIndexDic['redKing']])
            for redKingNdx in range(redKingsCoords.shape[0]):
                redKingCoords = [redKingsCoords[redKingNdx][1].item(),
                                 redKingsCoords[redKingNdx][2].item()]  # Index 0 is the dummy 'depth' dimension
                # NW
                firstDiagonal = [redKingCoords[0] - 1, redKingCoords[1] - 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0], firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('black'):
                    secondDiagonal = [redKingCoords[0] - 2, redKingCoords[1] - 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['NW'], 0, redKingCoords[0], redKingCoords[1]] = 1
                # NE
                firstDiagonal = [redKingCoords[0] - 1, redKingCoords[1] + 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('black'):
                    secondDiagonal = [redKingCoords[0] - 2, redKingCoords[1] + 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['NE'], 0, redKingCoords[0], redKingCoords[
                                1]] = 1
                # SE
                firstDiagonal = [redKingCoords[0] + 1, redKingCoords[1] + 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('black'):
                    secondDiagonal = [redKingCoords[0] + 2, redKingCoords[1] + 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['SE'], 0, redKingCoords[0], redKingCoords[
                                1]] = 1
                # SW
                firstDiagonal = [redKingCoords[0] + 1, redKingCoords[1] - 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('black'):
                    secondDiagonal = [redKingCoords[0] + 2, redKingCoords[1] - 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['SW'], 0, redKingCoords[0], redKingCoords[
                                1]] = 1
            # Red checkers
            redCheckersCoords = torch.nonzero(
                positionTensor[self.pieceToPositionPlaneIndexDic['redChecker']])
            for redCheckerNdx in range(redCheckersCoords.shape[0]):
                redCheckerCoords = [redCheckersCoords[redCheckerNdx][1].item(),
                                    redCheckersCoords[redCheckerNdx][2].item()]
                # SE
                firstDiagonal = [redCheckerCoords[0] + 1, redCheckerCoords[1] + 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('black'):
                    secondDiagonal = [redCheckerCoords[0] + 2, redCheckerCoords[1] + 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['SE'], 0, redCheckerCoords[0],
                            redCheckerCoords[1]] = 1
                # SW
                firstDiagonal = [redCheckerCoords[0] + 1, redCheckerCoords[1] - 1]
                firstDiagonalOccupation = self.SquareOccupation(positionTensor, firstDiagonal[0],
                                                                firstDiagonal[1])
                if firstDiagonalOccupation is not None and firstDiagonalOccupation.startswith('black'):
                    secondDiagonal = [redCheckerCoords[0] + 2, redCheckerCoords[1] - 2]
                    secondDiagonalOccupation = self.SquareOccupation(positionTensor, secondDiagonal[0],
                                                                     secondDiagonal[1])
                    if self.IsInCheckerboard(
                            secondDiagonal) and secondDiagonalOccupation is None:  # Possible capture
                        possibleCaptures[
                            self.moveDirectionToMovePlaneIndexDic['SW'], 0, redCheckerCoords[0],
                            redCheckerCoords[
                                1]] = 1
        return possibleCaptures


    def IsInCheckerboard(self, square):
        if square[0] >= 0 and square[0] <= 7 and square[1] >= 0 and square[1] <= 7:
            return True
        return False

def main():
    print ("checkers.py main()")
    authority = Authority()
    playersList = authority.PlayersList()
    positionTensor = torch.zeros(authority.PositionTensorShape())

    positionTensor[2, 0, 0, 1] = 1
    positionTensor[2, 0, 0, 7] = 1
    positionTensor[2, 0, 1, 0] = 1
    positionTensor[2, 0, 2, 5] = 1
    positionTensor[2, 0, 3, 4] = 1
    """positionTensor[2, 0, 2, 1] = 1
    positionTensor[2, 0, 2, 5] = 1
    positionTensor[2, 0, 2, 7] = 1
    positionTensor[2, 0, 3, 2] = 1"""

    positionTensor[1, 0, 4, 1] = 1
    positionTensor[1, 0, 5, 0] = 1
    positionTensor[1, 0, 5, 6] = 1
    positionTensor[1, 0, 6, 1] = 1
    positionTensor[1, 0, 6, 5] = 1
    positionTensor[1, 0, 7, 0] = 1
    positionTensor[1, 0, 7, 2] = 1
    positionTensor[1, 0, 7, 6] = 1

    positionTensor[0, 0, 0, 5] = 1
    #positionTensor[1, 0, 7, 6] = 1

    authority.Display(positionTensor)

    legalMovesMask = authority.LegalMovesMask(positionTensor, playersList[0])
    print ("legalMovesMask = \n{}".format(legalMovesMask))

    positionTensor, winner = authority.MoveWithString(positionTensor, playersList[0], '05-14')
    print ("After move:")
    authority.Display(positionTensor)
    print ("positionTensor = \n{}".format(positionTensor))
    """swappedPosition = authority.SwapPositions(positionTensor, playersList[0], playersList[1])
    print ("swappedPosition = \n{}".format(swappedPosition))"""
    """positionTensor, winner = authority.MoveWithString(positionTensor, playersList[1], '14-32')
    authority.Display(positionTensor)
    """


if __name__ == '__main__':
    main()