import ast
import torch
import collections # OrderedDict
import policy
import random
import numpy

class Net(torch.nn.Module):
    def __init__(self,
                 inputTensorSize,
                 bodyStructure,
                 outputTensorSize):  # Both input and output tensor sizes must be (C, D, H, W)
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

    def ChooseAMove(self, positionTensor, player, gameAuthority, preApplySoftMax=True, softMaxTemperature=1.0,
                    epsilon=0.1):
        actionValuesTensor = self.forward(positionTensor.unsqueeze(0)) # Add a dummy minibatch
        # Remove the dummy minibatch
        actionValuesTensor = torch.squeeze(actionValuesTensor, 0)

        chooseARandomMove = random.random() < epsilon
        if chooseARandomMove:
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

        runningSum = 0
        chosenCoordinates = None

        nonZeroCoordsTensor = torch.nonzero(legalMovesMask)
        for nonZeroCoordsNdx in range(nonZeroCoordsTensor.size(0)):
            nonZeroCoords = nonZeroCoordsTensor[nonZeroCoordsNdx]
            runningSum += normalizedActionValuesTensor[
                nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3]
            ]
            if runningSum >= randomNbr and chosenCoordinates is None:
                chosenCoordinates = (nonZeroCoords[0], nonZeroCoords[1], nonZeroCoords[2], nonZeroCoords[3])
                #print ("Net.ChooseAMove(): chosenCoordinates = {}".format(chosenCoordinates))
        if chosenCoordinates is None and randomNbr - runningSum < 0.000001: # Choose the last candidate
            chosenNdx = len(nonZeroCoordsTensor.size(0)) - 1
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

if __name__ == '__main__':
    main()