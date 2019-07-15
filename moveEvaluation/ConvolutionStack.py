import ast
import torch
import collections # OrderedDict
import policy
import random
import numpy
import os

class Net(torch.nn.Module):
    def __init__(self,
                 inputTensorSize=(1, 1, 1, 1),
                 bodyStructure=[(3, 1)],
                 outputTensorSize=(1, 1, 1, 1)):  # Both input and output tensor sizes must be (C, D, H, W)
        super(Net, self).__init__()
        if len(inputTensorSize) != 4:
            raise ValueError("Net.__init__(): The length of inputTensorSize ({}) is not 4 (C, D, H, W)".format(len(inputTensorSize)))
        if len(outputTensorSize) != 4:
            raise ValueError("Net.__init__(): The length of outputTensorSize ({}) is not 4 (C, D, H, W)".format(len(outputTensorSize)))
        self.inputTensorSize = inputTensorSize
        numberOfLayers = len(bodyStructure)
        if numberOfLayers == 0:
            raise ValueError("Net.__init__(): The list bodyStructure is empty")

        self.outputTensorSize = outputTensorSize

        bodyStructureDict = collections.OrderedDict()
        numberOfInputChannels = inputTensorSize[0]
        for layerNdx in range(numberOfLayers):
            kernelDimensionNumberPair = bodyStructure[layerNdx]
            if len(kernelDimensionNumberPair) != 2:
                raise ValueError("Net.__init__(): The length of the tuple {} is not 2".format(kernelDimensionNumberPair))
            layerName = 'layer_' + str(layerNdx)
            bodyStructureDict[layerName] = self.ConvolutionLayer(numberOfInputChannels,
                                                                 kernelDimensionNumberPair[0],
                                                                 kernelDimensionNumberPair[1])
            numberOfInputChannels = kernelDimensionNumberPair[1]

        self.bodyStructure = torch.nn.Sequential(bodyStructureDict)
        self.bodyActivationNumberOfEntries = bodyStructure[numberOfLayers - 1][1] * \
            inputTensorSize[1] * inputTensorSize[2] * inputTensorSize[3]
        #print ("Net.__init__(): self.bodyActivationNumberOfEntries = {}".format(self.bodyActivationNumberOfEntries))
        outputTensorNumberOfEntries = outputTensorSize[0] * outputTensorSize[1] * outputTensorSize[2] * outputTensorSize[3]
        self.fullyConnectedLayer = torch.nn.Linear(self.bodyActivationNumberOfEntries,
                                                   outputTensorNumberOfEntries)
        #self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.bodyStructureList = bodyStructure

    def ConvolutionLayer(self, inputNumberOfChannels, kernelDimension, numberOfOutputChannels):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=inputNumberOfChannels,
                            out_channels=numberOfOutputChannels,
                            kernel_size=kernelDimension,
                            padding=int(kernelDimension/2) ),
            torch.nn.BatchNorm3d(numberOfOutputChannels,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True),
            torch.nn.ReLU())

    def forward(self, inputTensor):
        minibatchSize = inputTensor.shape[0]
        bodyActivation = self.bodyStructure(inputTensor)
        #print ("Net.forward(): bodyActivation.shape = {}".format(bodyActivation.shape))
        bodyActivationVector = bodyActivation.view((minibatchSize, self.bodyActivationNumberOfEntries))
        linearActivation = self.fullyConnectedLayer(bodyActivationVector)
        squashedLinearActivation = -1.0 + 2.0 * self.sigmoid(linearActivation)#.tanh()
        #print ("Net.forward(): linearActivation.shape = {}".format(linearActivation.shape))
        outputTensor = squashedLinearActivation.view((minibatchSize,
                                              self.outputTensorSize[0],
                                              self.outputTensorSize[1],
                                              self.outputTensorSize[2],
                                              self.outputTensorSize[3],
                                              ))
        #channelsMatcherActivation = self.actionValuesChannelMatcher(bodyActivation)
        return outputTensor

    def ChooseAMove(self, positionTensor, player, gameAuthority, chooseHighestProbabilityIfAtLeast,
                    preApplySoftMax, softMaxTemperature,
                    epsilon):
        actionValuesTensor = self.forward(positionTensor.unsqueeze(0)) # Add a dummy minibatch
        # Remove the dummy minibatch
        actionValuesTensor = torch.squeeze(actionValuesTensor, 0)
        #print ("Net.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))

        chooseARandomMove = random.random() < epsilon
        if chooseARandomMove:
            #print ("Net.ChooseAMove(): Choosing a random move")
            return policy.ChooseARandomMove(positionTensor, player, gameAuthority)

        # Else: choose according to probabilities
        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)

        normalizedActionValuesTensor = policy.NormalizeProbabilities(actionValuesTensor,
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
            highestProbabilityCoords = self.CoordinatesFromFlatIndex(
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
            print ("Net.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
            print ("Net.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
            print ("Net.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
            print ("Net.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
            raise ValueError("Net.ChooseAMove(): legalMovesMask doesn't have a non-zero entry")

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

        """for ndx0 in range(actionValuesTensorShape[0]):
            for ndx1 in range(actionValuesTensorShape[1]):
                for ndx2 in range(actionValuesTensorShape[2]):
                    for ndx3 in range(actionValuesTensorShape[3]):
                        runningSum += normalizedActionValuesTensor[ndx0, ndx1, ndx2, ndx3]
                        if runningSum >= randomNbr and chosenCoordinates is None:
                            chosenCoordinates = (ndx0, ndx1, ndx2, ndx3)
        """

        if chosenCoordinates is None:
            print ("Net.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
            print ("Net.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
            print ("Net.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
            print ("Net.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
            print ("Net.ChooseAMove(): runningSum = {}; randomNbr = {}".format(runningSum, randomNbr))
            raise IndexError("Net.ChooseAMove(): chosenCoordinates is None...!???")

        chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
        chosenMoveArr[chosenCoordinates] = 1.0
        return torch.from_numpy(chosenMoveArr).float()

    def CoordinatesFromFlatIndex(self,
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

    def Save(self, directory, filenameSuffix):
        filename = 'Net_' + str(self.inputTensorSize) + '_' + \
                                str(self.bodyStructureList) + '_' + str(self.outputTensorSize) + '_' + \
                                filenameSuffix + '.pth'
        # Remove spaces
        filename = filename.replace(' ', '')
        filepath = os.path.join(directory, filename)
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath):
        filename = os.path.basename(filepath)
        # Tokenize the filename with '_'
        tokens = filename.split('_')
        if len(tokens) < 5:
            raise ValueError("Net.Load(): The number of tokens of {} is less than 5 ({})".format(filename, len(tokens)))
        inputTensorSize = ast.literal_eval(tokens[1])
        bodyStructure = ast.literal_eval(tokens[2])
        outputTensorSize = ast.literal_eval(tokens[3])
        self.__init__(inputTensorSize, bodyStructure, outputTensorSize)
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))

def main():

    print ("ConvolutionStack.py main()")
    inputTensorSize = (2, 1, 3, 4)
    outputTensorSize = (5, 1, 4, 7)
    neuralNet = Net(inputTensorSize, [(3, 32), (5, 16), (7, 8)], outputTensorSize)
    inputTensor = torch.randn((27, 2, 1, 3, 4))
    outputTensor = neuralNet(inputTensor)
    print ("main(): outputTensor.shape = {}".format(outputTensor.shape))

    """chosenMove = neuralNet.ChooseAMove(torch.randn(inputTensorSize),
                                       'player1',
                                       )
    """
    neuralNet.Save('./outputs', 'testSuffix')

    twinNet = Net()
    twinNet.Load("/home/sebastien/projects/DeepReinforcementLearning/outputs/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_18.pth")
    print ("main(): twinNet =\{}".format(twinNet))

if __name__ == '__main__':
    main()