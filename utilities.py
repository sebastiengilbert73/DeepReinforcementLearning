import argparse
import ast
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# To avoid `RuntimeError: received 0 items of ancdata` Cf. https://github.com/pytorch/pytorch/issues/973
import math
import random
import sys
import statistics
import numpy

def NormalizeProbabilities(moveProbabilitiesTensor, legalMovesMask, preApplySoftMax=True, softMaxTemperature=1.0):
    if moveProbabilitiesTensor.shape != legalMovesMask.shape:
        raise ValueError("utilities.NormalizeProbabilities(): The shape of moveProbabilitiesTensor ({}) doesn't match the shape of legalMovesMask ({})".format(moveProbabilitiesTensor, legalMovesMask))
    # Make a copy to avoid changing moveProbabilitiesTensor
    moveProbabilitiesCopyTensor = torch.zeros(moveProbabilitiesTensor.shape)
    moveProbabilitiesCopyTensor.copy_(moveProbabilitiesTensor)
    # Flatten the tensors
    moveProbabilitiesVector = moveProbabilitiesCopyTensor.view(moveProbabilitiesCopyTensor.numel())
    legalMovesVector = legalMovesMask.view(legalMovesMask.numel())
    legalProbabilitiesValues = []

    for index in range(moveProbabilitiesVector.shape[0]):
        if legalMovesVector[index] == 1:
            legalProbabilitiesValues.append(moveProbabilitiesVector[index])

    if preApplySoftMax:
        legalProbabilitiesVector = torch.softmax(torch.Tensor(legalProbabilitiesValues)/softMaxTemperature, 0)
        runningNdx = 0
        for index in range(moveProbabilitiesVector.shape[0]):
            if legalMovesVector[index] == 0:
                moveProbabilitiesVector[index] = 0
            else:
                moveProbabilitiesVector[index] = legalProbabilitiesVector[runningNdx]
                runningNdx += 1
    else: # Normalize
        sum = 0
        for index in range(moveProbabilitiesVector.shape[0]):
            if moveProbabilitiesVector[index] < 0:
                raise ValueError("utilities.NormalizeProbabilities(): The probability value {} is negative".format(moveProbabilitiesVector[index]))
            sum += moveProbabilitiesVector[index]
        moveProbabilitiesVector = moveProbabilitiesVector/sum

    # Resize to original size
    normalizedProbabilitiesTensor = moveProbabilitiesVector.view(moveProbabilitiesCopyTensor.shape)
    return normalizedProbabilitiesTensor

def ChooseARandomMove(positionTensor, player, gameAuthority):

    legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
    numberOfLegalMoves = torch.nonzero(legalMovesMask).size(0)
    if numberOfLegalMoves == 0:
        if gameAuthority.RaiseAnErrorIfNoLegalMove():
            raise ValueError("utilities.ChooseARandomMove(): The number of legal moves is zero. player = {}; positionTensor = \n{}".format(player, positionTensor))
        return None
    randomNbr = random.randint(1, numberOfLegalMoves)
    probabilitiesTensorShape = legalMovesMask.shape
    runningSum = 0
    chosenCoordinates = None
    for ndx0 in range(probabilitiesTensorShape[0]):
        for ndx1 in range(probabilitiesTensorShape[1]):
            for ndx2 in range(probabilitiesTensorShape[2]):
                for ndx3 in range(probabilitiesTensorShape[3]):
                    runningSum += legalMovesMask[ndx0, ndx1, ndx2, ndx3]
                    if runningSum >= randomNbr and chosenCoordinates is None:
                        chosenCoordinates = (ndx0, ndx1, ndx2, ndx3)

    if chosenCoordinates is None:
        raise IndexError("utilities.ChooseARandomMove(): choseCoordinates is None...!???")

    chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
    chosenMoveArr[chosenCoordinates] = 1.0
    return torch.from_numpy(chosenMoveArr).float()

def MinibatchIndices(numberOfSamples, minibatchSize):
	shuffledList = numpy.arange(numberOfSamples)
	numpy.random.shuffle(shuffledList)
	minibatchesIndicesList = []
	numberOfWholeLists = int(numberOfSamples / minibatchSize)
	for wholeListNdx in range(numberOfWholeLists):
		minibatchIndices = shuffledList[ wholeListNdx * minibatchSize : (wholeListNdx + 1) * minibatchSize ]
		minibatchesIndicesList.append(minibatchIndices)
	# Add the last incomplete minibatch
	if numberOfWholeLists * minibatchSize < numberOfSamples:
		lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
		minibatchesIndicesList.append(lastMinibatchIndices)
	return minibatchesIndicesList

def MinibatchTensor(positionsList):
    if len(positionsList) == 0:
        raise ArgumentException("utilities.MinibatchTensor(): Empty list of positions")
    #print ("MinibatchTensor(): len(positionsList) = {}; positionsList[0].shape = {}".format(len(positionsList), positionsList[0].shape) )
    positionShape = positionsList[0].shape
    minibatchTensor = torch.zeros(len(positionsList), positionShape[0],
                                  positionShape[1], positionShape[2], positionShape[3]) # NCDHW
    for n in range(len(positionsList)):
        #print ("MinibatchTensor(): positionsList[n].shape = {}".format(positionsList[n].shape))
        #print ("MinibatchTensor(): positionsList[n] = {}".format(positionsList[n]))
        minibatchTensor[n] = positionsList[n]
    return minibatchTensor

def adjust_lr(optimizer, desiredLearningRate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = desiredLearningRate

def CoordinatesFromFlatIndex(
            flatIndex,
            tensorShape):
    numberOfEntries = tensorShape[0] * tensorShape[1] * \
        tensorShape[2] * tensorShape[3]

    flatTensor = torch.zeros((numberOfEntries))
    flatTensor[flatIndex] = 1
    oneHotTensor = flatTensor.view(tensorShape)
    coordsTensor = oneHotTensor.nonzero()[0]
    coords = (coordsTensor[0].item(),
              coordsTensor[1].item(),
              coordsTensor[2].item(),
              coordsTensor[3].item())
    return coords

def StandardDeviationOfLegalValues(
        moveValuesTensor,
        legalMovesMask
        ):
    legalValues = []
    legalCoords = legalMovesMask.nonzero()
    #print ("StandardDeviationOfLegalValues(): legalCoords = {}".format(legalCoords))
    for valueNdx in range(legalCoords.shape[0]):
        channel = legalCoords[valueNdx][-4]
        depth = legalCoords[valueNdx][-3]
        row = legalCoords[valueNdx][-2]
        col = legalCoords[valueNdx][-1]
        #print ("StandardDeviationOfLegalValues(): legalCoords[valueNdx].shape = {}".format(legalCoords[valueNdx].shape))
        if legalCoords[valueNdx].shape[0] == 5:
            sampleNdx = legalCoords[valueNdx][-5]
            legalValues.append(moveValuesTensor[sampleNdx][channel][depth][row][col].item())
        #print ("StandardDeviationOfLegalValues(): moveValuesTensor[channel][depth][row][col] = {}".format(moveValuesTensor[channel][depth][row][col]))
        else:
            legalValues.append(moveValuesTensor[channel][depth][row][col].item())
    #print ("StandardDeviationOfLegalValues(): legalValues = {}".format(legalValues))
    if len(legalValues) > 1:
        stdDev = statistics.stdev(legalValues)
    else:
        stdDev = 0
    #print ("StandardDeviationOfLegalValues(): stdDev = {}".format(stdDev))
    return stdDev

def ChooseAMove(neuralNetwork, positionTensor, player, gameAuthority, chooseHighestProbabilityIfAtLeast,
                preApplySoftMax, softMaxTemperature,
                epsilon):
    actionValuesTensor = neuralNetwork(positionTensor.unsqueeze(0)) # Add a dummy minibatch
    # Remove the dummy minibatch
    actionValuesTensor = torch.squeeze(actionValuesTensor, 0)
    #print ("Net.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))

    chooseARandomMove = random.random() < epsilon
    if chooseARandomMove:
        #print ("Net.ChooseAMove(): Choosing a random move")
        return ChooseARandomMove(positionTensor, player, gameAuthority)

    # Else: choose according to probabilities
    legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
    if torch.nonzero(legalMovesMask).size(0) == 0:
        return None

    normalizedActionValuesTensor = NormalizeProbabilities(actionValuesTensor,
                                                           legalMovesMask,
                                                           preApplySoftMax=preApplySoftMax,
                                                           softMaxTemperature=softMaxTemperature)

    #print ("Net.ChooseAMove(): normalizedActionValuesTensor = \n{}".format(normalizedActionValuesTensor))
    randomNbr = random.random()
    actionValuesTensorShape = normalizedActionValuesTensor.shape

    maximumProbabilityFlatIndex = normalizedActionValuesTensor.argmax().item()
    maximumProbability = normalizedActionValuesTensor.view(-1)[
        maximumProbabilityFlatIndex].item()
    if maximumProbability >= chooseHighestProbabilityIfAtLeast:
        #print ("Net.ChooseAMove(): The maximum probability ({}) is above the threshold ({})".format(maximumProbability, chooseHighestProbabilityIfAtLeast))
        highestProbabilityCoords = CoordinatesFromFlatIndex(
            maximumProbabilityFlatIndex,
            actionValuesTensorShape
        )
        chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
        chosenMoveArr[highestProbabilityCoords] = 1.0
        return torch.from_numpy(chosenMoveArr).float()

    # Else: Choose with roulette
    #print ("Net.ChooseAMove(): Roulette!")
    runningSum = 0
    chosenCoordinates = None

    nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
    if nonZeroCoordsTensor.size(0) == 0:
        print ("utilities.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
        print ("utilities.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
        print ("utilities.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
        print ("utilities.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
        raise ValueError("utilities.ChooseAMove(): legalMovesMask doesn't have a non-zero entry")

    for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0) - 1):
        nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
        runningSum += normalizedActionValuesTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]
        ]
        if runningSum >= randomNbr and chosenCoordinates is None:
            chosenCoordinates = (nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3])
            #print ("Net.ChooseAMove(): chosenCoordinates = {}".format(chosenCoordinates))
            break # Stop looping
    if chosenCoordinates is None:# and randomNbr - runningSum < 0.000001: # Choose the last candidate
        chosenNdx = nonZeroCoordsTensor.size(0) - 1
        nonZeroCoords = nonZeroCoordsTensor[chosenNdx]
        chosenCoordinates = (nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3])

    if chosenCoordinates is None:
        print ("utilities.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
        print ("utilities.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
        print ("utilities.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
        print ("utilities.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
        print ("utilities.ChooseAMove(): runningSum = {}; randomNbr = {}".format(runningSum, randomNbr))
        raise IndexError("utilities.ChooseAMove(): chosenCoordinates is None...!???")

    chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
    chosenMoveArr[chosenCoordinates] = 1.0
    return torch.from_numpy(chosenMoveArr).float()

def LegalMoveTensorsList(gameAuthority, positionTensor, nextPlayer):
    moveTensorShape = gameAuthority.MoveTensorShape()
    legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, nextPlayer)
    nonZeroCoordsTensor = legalMovesMask.nonzero()
    legalMoveTensorsList = []
    for candidateMoveNdx in range(nonZeroCoordsTensor.shape[0]):
        candidateMoveTensor = torch.zeros(moveTensorShape)
        nonZeroCoords = nonZeroCoordsTensor[candidateMoveNdx]
        candidateMoveTensor[
            nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]] = 1.0
        legalMoveTensorsList.append(candidateMoveTensor)
    return legalMoveTensorsList

def LegalCandidatePositionsAfterMove(gameAuthority, positionTensor, nextPlayer):
    legalMovesTensorList = LegalMoveTensorsList(gameAuthority, positionTensor, nextPlayer)
    legalCandidatePositionsAfterMove = []
    for legalMoveTsr in legalMovesTensorList:
        (candiatePositionTensor, winner) = gameAuthority.Move(positionTensor, nextPlayer, legalMoveTsr)
        legalCandidatePositionsAfterMove.append((candiatePositionTensor, winner))
    return legalCandidatePositionsAfterMove

def LegalCandidatePositionsAfterMoveDictionary(gameAuthority, positionTensor, nextPlayer):
    legalMovesTensorList = LegalMoveTensorsList(gameAuthority, positionTensor, nextPlayer)
    legalCandidatePositionsAfterMoveToWinnerDict = {}
    for legalMoveTsr in legalMovesTensorList:
        (candidatePositionTensor, winner) = gameAuthority.Move(positionTensor, nextPlayer, legalMoveTsr)
        legalCandidatePositionsAfterMoveToWinnerDict[candidatePositionTensor] = winner
        #print ("LegalCandidatePositionsAfterMoveDictionary(): candidatePositionTensor.shape = {}".format(candidatePositionTensor.shape))
    #print ("LegalCandidatePositionsAfterMoveDictionary(): legalCandidatePositionsAfterMoveToWinnerDict = {}".format(legalCandidatePositionsAfterMoveToWinnerDict))
    return legalCandidatePositionsAfterMoveToWinnerDict

if __name__ == '__main__':
    import checkers

    authority = checkers.Authority()
