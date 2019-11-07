import ast
import torch
import collections # OrderedDict
import random
import numpy
import os

class Net(torch.nn.Module):
    def __init__(self,
                 positionTensorShape=(1, 1, 1, 1), # Both position tensor size must be (C, D, H, W)
                 bodyStructure=[(3, 32, 2), (3, 64, 2)], # (kernelSize, numberOfChannels, stride)
                 numberOfLatentVariables=100,
                 ):
        super(Net, self).__init__()
        if len(positionTensorShape) != 4:
            raise ValueError("Net.__init__(): The length of positionTensorShape ({}) is not 4 (C, D, H, W)".format(len(positionTensorShape)))
        self.positionTensorShape = positionTensorShape
        self.numberOfLatentVariables = numberOfLatentVariables

        numberOfLayers = len(bodyStructure)
        if numberOfLayers == 0:
            raise ValueError("Net.__init__(): The list bodyStructure is empty")

        bodyStructureDict = collections.OrderedDict()
        self.layerNameToTensorPhysicalShapeDict = {}
        self.encodingLayerNames = []
        previousLayerPhysicalShape = (self.positionTensorShape[1], self.positionTensorShape[2], self.positionTensorShape[3])
        numberOfInputChannels = positionTensorShape[0]
        #strideProduct = 1
        for layerNdx in range(numberOfLayers):
            kernelDimensionNumberPair = bodyStructure[layerNdx]
            if len(kernelDimensionNumberPair) != 2 and len(kernelDimensionNumberPair) != 3:
                raise ValueError("Net.__init__(): The length of the tuple {} is not 2 or 3".format(kernelDimensionNumberPair))
            layerName = 'layer_' + str(layerNdx)
            stride = 1
            if len(kernelDimensionNumberPair) == 3:
                stride = kernelDimensionNumberPair[2]


            self.layerNameToTensorPhysicalShapeDict[layerName] = ( max((previousLayerPhysicalShape[0] + 1)//stride, 1),
                                                                   max((previousLayerPhysicalShape[1] + 1)//stride, 1),
                                                                   max((previousLayerPhysicalShape[2] + 1)//stride, 1))
            #print ("self.layerNameToTensorPhysicalShapeDict[{}] = {}".format(layerName, self.layerNameToTensorPhysicalShapeDict[layerName]))
            self.layerNameToTensorPhysicalShapeDict['decoding_' + str(layerNdx + 1)] = self.layerNameToTensorPhysicalShapeDict[layerName]
            previousLayerPhysicalShape = self.layerNameToTensorPhysicalShapeDict[layerName]
            #strideProduct = strideProduct * stride**2
            #if positionTensorShape[1] > 1: # If there is depth
            #    strideProduct = strideProduct * stride
            bodyStructureDict[layerName] = self.ConvolutionLayer(numberOfInputChannels,
                                                                 kernelDimensionNumberPair[0],
                                                                 kernelDimensionNumberPair[1],
                                                                 stride,
                                                                 dilation=1)
            self.encodingLayerNames.append(layerName)
            numberOfInputChannels = kernelDimensionNumberPair[1]
        self.bodyStructureSeq = torch.nn.Sequential(bodyStructureDict)

        self.bodyActivationNumberOfEntries = bodyStructure[numberOfLayers - 1][1] * \
                                             previousLayerPhysicalShape[0] * previousLayerPhysicalShape[1] * previousLayerPhysicalShape[2]

        self.fullyConnectedLayer = torch.nn.Linear(self.bodyActivationNumberOfEntries,
                                                   numberOfLatentVariables)
        self.bodyStructureList = bodyStructure
        self.sigmoid = torch.nn.Sigmoid()

        self.decodingFullyConnectedLayer = torch.nn.Linear(numberOfLatentVariables, self.bodyActivationNumberOfEntries)
        #print ("__init__(): self.bodyActivationNumberOfEntries = {}".format(self.bodyActivationNumberOfEntries))
        # Deconvolutions
        decodingConvTransposedDict = collections.OrderedDict()
        #numberOfInputChannels = bodyStructure[-1][1] # The number of channels of the last convolution layer
        for layerNdx in reversed(range(numberOfLayers)):
            numberOfInputChannels = bodyStructure[layerNdx][1]
            kernelSize = bodyStructure[layerNdx][0]
            layerName = 'decoding_' + str(layerNdx)
            if layerNdx == 0:
                outputNumberOfChannels = positionTensorShape[0]
            else:
                outputNumberOfChannels = bodyStructure[layerNdx - 1][1]
            stride = 1
            if len(bodyStructure[layerNdx]) == 3:
                stride = bodyStructure[layerNdx][2]
            #decodingDict[layerName] = self.ConvolutionLayer(numberOfInputChannels, kernelSize, outputNumberOfChannels, stride=1, dilation=dilation)
            decodingConvTransposedDict[layerName] = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(in_channels=numberOfInputChannels,
                            out_channels=outputNumberOfChannels,
                            kernel_size=kernelSize,
                            padding=kernelSize//2,
                            stride=stride,
                            dilation=1,
                            output_padding=1),
                torch.nn.BatchNorm3d(outputNumberOfChannels,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True),
                torch.nn.ReLU())

        self.decodingSeq = torch.nn.Sequential(decodingConvTransposedDict)

    def ConvolutionLayer(self, inputNumberOfChannels, kernelDimension, numberOfOutputChannels, stride=1, dilation=1):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=inputNumberOfChannels,
                            out_channels=numberOfOutputChannels,
                            kernel_size=kernelDimension,
                            padding=int(kernelDimension/2),
                            stride=stride,
                            dilation=dilation),
            torch.nn.BatchNorm3d(numberOfOutputChannels,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True),
            torch.nn.ReLU())

    def forward(self, inputTensor):
        minibatchSize = inputTensor.shape[0]
        """bodyActivation = self.bodyStructure(inputTensor)
        #print ("Net.forward(): bodyActivation.shape = {}; self.bodyActivationNumberOfEntries = {}".format(bodyActivation.shape, self.bodyActivationNumberOfEntries))
        bodyActivationVector = bodyActivation.view((minibatchSize, self.bodyActivationNumberOfEntries))
        linearActivation = self.fullyConnectedLayer(bodyActivationVector)
        squashedLinearActivation = -1.0 + 2.0 * self.sigmoid(linearActivation)

        # Decoding
        #print ("forward(): self.positionTensorShape = {}".format(self.positionTensorShape))
        #print ("forward(): squashedLinearActivation.shape = {}".format(squashedLinearActivation.shape))
        decodingActivation = self.decodingFullyConnectedLayer(squashedLinearActivation)
        print ("forward(): decodingActivation.shape = {}".format(decodingActivation.shape))
        #print ("forward(): self.bodyStructureList[-1][1] = {}".format(self.bodyStructureList[-1][1]))
        decodingActivation = decodingActivation.view(minibatchSize,
                                                           self.bodyStructureList[-1][1],
                                                           self.positionTensorShape[1],
                                                           self.positionTensorShape[2],
                                                           self.positionTensorShape[3])
        decodingActivation = self.decoding(decodingActivation)
        return decodingActivation
        """
        activationTensor = inputTensor
        #for (layerName, convLayer) in self.bodyStructureDict.items():
        #    activationTensor = convLayer(activationTensor)
        #    #print ("forward(): After {}, activationTensor.shape = {}".format(layerName, activationTensor.shape))
        activationTensor = self.bodyStructureSeq(activationTensor)

        activationTensor = activationTensor.view(minibatchSize, self.bodyActivationNumberOfEntries)
        activationTensor = self.fullyConnectedLayer(activationTensor)
        #print ("forward(): After fullyConnectedLayer, activationTensor.shape = {}".format(activationTensor.shape))
        activationTensor = self.sigmoid(activationTensor)

        # Decoding
        activationTensor = self.decodingFullyConnectedLayer(activationTensor)
        lastEncodingLayerName = self.encodingLayerNames[-1]
        lastLayerShape = self.layerNameToTensorPhysicalShapeDict[lastEncodingLayerName]
        #print ("forward(): lastLayerShape = {}".format(lastLayerShape))
        lastLayerNumberOfChannels = self.bodyStructureList[-1][1]
        #print ("forward(): lastLayerNumberOfChannels = {}".format(lastLayerNumberOfChannels))
        activationTensor = activationTensor.view(minibatchSize, lastLayerNumberOfChannels, lastLayerShape[0], lastLayerShape[1], lastLayerShape[2])
        #print ("forward(): After .view(): activationTensor.shape = {}".format(activationTensor.shape))
        activationTensor = self.decodingSeq(activationTensor)
        #print ("forward(): After decodingSeq(), activationTensor.shape = {}".format(activationTensor.shape))
        # Get rid of spurrious physical dimensions
        activationTensor = activationTensor[:, :, 0:self.positionTensorShape[1], 0:self.positionTensorShape[2], 0:self.positionTensorShape[3] ]
        #print ("forward(): After pruning, activationTensor.shape = {}".format(activationTensor.shape))
        """for (layerName, convTransposedLayer) in self.decodingConvTransposedDict.items():
            activationTensor = convTransposedLayer(activationTensor)
            #print ("forward(): After {}, activationTensor.shape = {}".format(layerName, activationTensor.shape))
            # Get rid of spurrious dimensions
            if layerName.endswith('_0'):
                desiredPhysicalShape = (self.positionTensorShape[1], self.positionTensorShape[2], self.positionTensorShape[3])
            else:
                desiredPhysicalShape = self.layerNameToTensorPhysicalShapeDict[layerName]
            activationTensor = activationTensor[:, :, 0:desiredPhysicalShape[0], 0:desiredPhysicalShape[1], 0:desiredPhysicalShape[2]]
            #print ("forward(): After getting rid of spurrious dimensions, activationTensor.shape = {}".format(activationTensor.shape))
        """
        return activationTensor# self.sigmoid(activationTensor)

    def Save(self, directory, filenameSuffix):
        filename = 'AutoencoderNet_' + str(self.positionTensorShape) + '_' + \
                                str(self.bodyStructureList) + '_' + str(self.numberOfLatentVariables) + '_' + \
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
        positionTensorShape = ast.literal_eval(tokens[1])
        bodyStructureList = ast.literal_eval(tokens[2])
        numberOfLatentVariables = int(tokens[3])
        self.__init__(positionTensorShape, bodyStructureList, numberOfLatentVariables)
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))
        self.eval()

    def AddConvolutionLayer(self, kernelDimension):
        lastLayer = self.bodyStructure[-1]
        print ("AddConvolutionLayer(): lastLayer = {}".format(lastLayer))
        print ("AddConvolutionLayer(): lastLayer[0] = {}".format(lastLayer[0]))
        print ("AddConvolutionLayer(): lastLayer[0].in_channels = {}".format(lastLayer[0].in_channels))
        #lastLayerNumberOfInputChannels = lastLayer[0].in_channels
        lastLayerNumberOfOutputChannels = lastLayer[0].out_channels
        nextConvolutionNdx = len(self.bodyStructure)
        additionalConvolution = self.ConvolutionLayer(lastLayerNumberOfOutputChannels, kernelDimension, lastLayerNumberOfOutputChannels)
        #decodingAdditionalConvolution = self.ConvolutionLayer(lastLayerNumberOfOutputChannels, kernelDimension, lastLayerNumberOfInputChannels)

        self.bodyStructure.add_module('layer_' + str(nextConvolutionNdx), additionalConvolution)
        self.bodyStructureList.append((kernelDimension, lastLayerNumberOfOutputChannels))

        # Deconvolutions
        decodingDict = collections.OrderedDict()
        for layerNdx in reversed(range(len(self.bodyStructureList))):
            numberOfInputChannels = self.bodyStructureList[layerNdx][1]
            kernelSize = self.bodyStructureList[layerNdx][0]
            layerName = 'decoding_' + str(layerNdx)
            if layerNdx == 0:
                outputNumberOfChannels = self.positionTensorShape[0]
            else:
                outputNumberOfChannels = self.bodyStructureList[layerNdx - 1][1]
            decodingDict[layerName] = self.ConvolutionLayer(numberOfInputChannels, kernelSize, outputNumberOfChannels)
        self.decoding = torch.nn.Sequential(decodingDict)


def ConvertToOneHotPositionTensor(reconstructedPositionTensor):
    positionTensorShape = reconstructedPositionTensor.shape #([m,] C, D, H, W)
    oneHotPositionTensor = torch.zeros(positionTensorShape, dtype=torch.int64)
    thereIsAMinibatch = False
    if len(positionTensorShape) == 5:
        thereIsAMinibatch = True
    elif len (positionTensorShape) != 4:
        raise ValueError("ConvertToOneHotPositionTensor(): The length of the input tensor shape ({}) is not 4 or 5".format(len (positionTensorShape)))

    if thereIsAMinibatch:
        for minibatchNdx in range(positionTensorShape[0]):
            for depth in range(positionTensorShape[2]):
                for row in range(positionTensorShape[3]):
                    for column in range(positionTensorShape[4]):
                        highestValue = -1.0
                        highestChannelNdx = -1
                        for channelNdx in range(positionTensorShape[1]):
                            value = reconstructedPositionTensor[minibatchNdx][channelNdx][depth][row][column]
                            if value > highestValue:
                                highestValue = value
                                highestChannelNdx = channelNdx
                        if highestValue >= 0.5:
                            oneHotPositionTensor[minibatchNdx][highestChannelNdx][depth][row][column] = 1
    else:
        for depth in range(positionTensorShape[1]):
            for row in range(positionTensorShape[2]):
                for column in range(positionTensorShape[3]):
                    highestValue = -1.0
                    highestChannelNdx = -1
                    for channelNdx in range(positionTensorShape[0]):
                        value = reconstructedPositionTensor[channelNdx][depth][row][column]
                        if value > highestValue:
                            highestValue = value
                            highestChannelNdx = channelNdx
                    if highestValue >= 0.5:
                        oneHotPositionTensor[highestChannelNdx][depth][row][column] = 1

    return oneHotPositionTensor

class Banane(torch.nn.Module):
    def __init__(self,
                ):
        super(Banane, self).__init__()
        self.positionTensorShape = (6, 1, 8, 8)
        self.numberOfLatentVariables = 100
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=6,
                            out_channels=32,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            dilation=1),
            torch.nn.BatchNorm3d(32,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True),
            torch.nn.ReLU())
        self.linear1 = torch.nn.Linear(32*1*4*4, self.numberOfLatentVariables)
        self.sigmoid = torch.nn.Sigmoid()
        self.decodingLinear = torch.nn.Linear(self.numberOfLatentVariables, 32*1*4*4)
        self.DecodingConv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=32,
                            out_channels=6,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            dilation=1,
                            output_padding=1),
            torch.nn.BatchNorm3d(6,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True),
            torch.nn.ReLU())

    def forward(self, inputTensor):
        minibatchSize = inputTensor.shape[0]
        afterConvTensor = self.conv1(inputTensor)
        #print ("forward(): afterConvTensor.shape = {}".format(afterConvTensor.shape))
        linearisedTensor = afterConvTensor.view(minibatchSize, 32*1*4*4)
        #print ("forward(): linearisedTensor.shape = {}".format(linearisedTensor.shape))
        linearisedTensor = self.linear1(linearisedTensor)
        #print ("forward(): After linear1, linearisedTensor.shape = {}".format(linearisedTensor.shape))
        linearisedTensor = self.sigmoid(linearisedTensor)
        #print ("forward(): After sigmoid, linearisedTensor.shape = {}".format(linearisedTensor.shape))
        linearisedTensor = self.decodingLinear(linearisedTensor)
        #print ("forward(): After decodingLinear, linearisedTensor.shape = {}".format(linearisedTensor.shape))
        decodedTensor = linearisedTensor.view(minibatchSize, 32, 1, 4, 4)
        #print ("forward(): decodedTensor.shape = {}".format(decodedTensor.shape))
        decodedTensor = self.DecodingConv1(decodedTensor)
        #print ("forward(): decodedTensor.shape = {}".format(decodedTensor.shape))
        # Get rid of spurrious dimensions
        decodedTensor = decodedTensor[:, :, 0:self.positionTensorShape[1], 0:self.positionTensorShape[2], 0:self.positionTensorShape[3]]
        #print ("forward(): After pruning: decodedTensor.shape = {}".format(decodedTensor.shape))
        return self.sigmoid(decodedTensor)


def main():
    print ("autoencoder > position.py main()")

    import checkers
    authority = checkers.Authority()
    positionTensorShape = authority.PositionTensorShape()
    autoencoder = Net((6, 1, 8, 8), [(3, 32, 2), (3, 64, 2)], 100)

    inputPositionTensor = authority.InitialPosition()
    outputTensor = autoencoder(inputPositionTensor.unsqueeze(0))

if __name__ == '__main__':
    main()