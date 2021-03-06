import ast
import torch
import collections # OrderedDict
import random
import numpy
import os
import math
import sys

class Net(torch.nn.Module):
    def __init__(self,
                 positionTensorShape=(1, 1, 1, 1), # The position tensor size must be (C, D, H, W)
                 bodyStructure=[(3, 32, 2), (3, 64, 2)], # (kernelSize, numberOfChannels, stride)
                 numberOfLatentVariables=100,
                 zeroPadding=True
                 ):
        super(Net, self).__init__()
        if len(positionTensorShape) != 4:
            raise ValueError("Net.__init__(): The length of positionTensorShape ({}) is not 4 (C, D, H, W)".format(len(positionTensorShape)))
        self.positionTensorShape = positionTensorShape
        self.bodyStructureList = bodyStructure
        self.numberOfLatentVariables = numberOfLatentVariables
        self.zeroPadding = zeroPadding

        numberOfLayers = len(bodyStructure)
        if numberOfLayers == 0:
            raise ValueError("Net.__init__(): The list bodyStructure is empty")

        bodyStructureDict = collections.OrderedDict()
        self.layerNameToTensorPhysicalShapeDict = {}
        self.encodingLayerNames = []
        previousLayerPhysicalShape = (self.positionTensorShape[1], self.positionTensorShape[2], self.positionTensorShape[3])
        numberOfInputChannels = positionTensorShape[0]

        for layerNdx in range(numberOfLayers):
            kernelDimensionNumberPair = bodyStructure[layerNdx]
            if len(kernelDimensionNumberPair) != 2 and len(kernelDimensionNumberPair) != 3:
                raise ValueError("Net.__init__(): The length of the tuple {} is not 2 or 3".format(kernelDimensionNumberPair))
            layerName = 'layer_' + str(layerNdx)
            stride = 1
            if len(kernelDimensionNumberPair) == 3:
                stride = kernelDimensionNumberPair[2]

            #convErosion = 0
            if zeroPadding:
                convErosion = 0
            else:
                convErosion = kernelDimensionNumberPair[0] - 1
            #print ("Net.__init__(): convErosion = {}".format(convErosion))
            if stride == 1:
                self.layerNameToTensorPhysicalShapeDict[layerName] = (max(previousLayerPhysicalShape[0] - convErosion, 1),
                                                                      max(previousLayerPhysicalShape[1] - convErosion, 1),
                                                                      max(previousLayerPhysicalShape[2] - convErosion, 1))
            else:
                self.layerNameToTensorPhysicalShapeDict[layerName] = ( max((previousLayerPhysicalShape[0] - convErosion + 1)//stride, 1),
                                                                   max((previousLayerPhysicalShape[1] - convErosion + 1)//stride, 1),
                                                                   max((previousLayerPhysicalShape[2] - convErosion + 1)//stride, 1))
            #print ("self.layerNameToTensorPhysicalShapeDict[{}] = {}".format(layerName, self.layerNameToTensorPhysicalShapeDict[layerName]))


            kernelDimension = ( min(kernelDimensionNumberPair[0], previousLayerPhysicalShape[0]),
                                min(kernelDimensionNumberPair[0], previousLayerPhysicalShape[1]),
                                min(kernelDimensionNumberPair[0], previousLayerPhysicalShape[2]))
            bodyStructureDict[layerName] = ConvolutionLayer(numberOfInputChannels,
                                                                 kernelDimension,
                                                                 kernelDimensionNumberPair[1],
                                                                 stride,
                                                                 dilation=1,
                                                                 zeroPadding=zeroPadding)
            self.encodingLayerNames.append(layerName)
            numberOfInputChannels = kernelDimensionNumberPair[1]
            previousLayerPhysicalShape = self.layerNameToTensorPhysicalShapeDict[layerName]
        #print ("Net.__init__(): bodyStructureDict = {}".format(bodyStructureDict))
        #print("Net.__init__(): previousLayerPhysicalShape = {}".format(previousLayerPhysicalShape))
        self.bodyStructureSeq = torch.nn.Sequential(bodyStructureDict)
        self.bodyActivationNumberOfEntries = bodyStructure[numberOfLayers - 1][1] * \
                                             previousLayerPhysicalShape[0] * previousLayerPhysicalShape[1] * previousLayerPhysicalShape[2]
        #print("Net.__init__(): self.bodyActivationNumberOfEntries = {}".format(self.bodyActivationNumberOfEntries))
        self.fullyConnectedLayer = torch.nn.Linear(self.bodyActivationNumberOfEntries,
                                                   numberOfLatentVariables)
        self.decodingFullyConnectedLayer = torch.nn.Linear(numberOfLatentVariables, self.bodyActivationNumberOfEntries)
        self.ReLU = torch.nn.ReLU()

        # Decoder: 2 layers with an intermediate number of neurons given by sqrt(N1 * N2)
        intermediateNumberOfNeurons = int (math.sqrt(self.bodyActivationNumberOfEntries * self.positionTensorShape[0] * \
                                                self.positionTensorShape[1] * self.positionTensorShape[2] * \
                                                self.positionTensorShape[3]) )
        self.decodingLinear1 = torch.nn.Linear(self.numberOfLatentVariables, intermediateNumberOfNeurons)
        self.decodingLinear2 = torch.nn.Linear(intermediateNumberOfNeurons, self.positionTensorShape[0] * \
                                                self.positionTensorShape[1] * self.positionTensorShape[2] * \
                                                self.positionTensorShape[3])


    def forward(self, inputTensor):
        """
        minibatchSize = inputTensor.shape[0]
        activationTensor = inputTensor
        activationTensor = self.bodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.bodyActivationNumberOfEntries)
        activationTensor = self.fullyConnectedLayer(activationTensor)
        activationTensor = self.ReLU(activationTensor)
        """
        encodedTsr = self.Encode(inputTensor)

        # Decoding
        """activationTensor = self.decodingLinear1 (activationTensor)
        activationTensor = self.ReLU(activationTensor)
        activationTensor = self.decodingLinear2(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.positionTensorShape[0] , \
                                                self.positionTensorShape[1] , self.positionTensorShape[2] , \
                                                self.positionTensorShape[3])
        return activationTensor # No Sigmoid: If an entry is negative in this activation tensor, the corresponding
                                # entry in the high level tensor will be a 0. Otherwise, it will be a 1. Cf. ConvertToOneHotPositionTensor(*)
        """
        return self.Decode(encodedTsr)

    def Encode(self, inputTensor):
        minibatchSize = inputTensor.shape[0]

        activationTensor = inputTensor
        activationTensor = self.bodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.bodyActivationNumberOfEntries)
        activationTensor = self.fullyConnectedLayer(activationTensor)
        activationTensor = self.ReLU(activationTensor)
        return activationTensor

    def Decode(self, encodedTsr):
        minibatchSize = encodedTsr.shape[0]
        activationTensor = self.decodingLinear1(encodedTsr)
        activationTensor = self.ReLU(activationTensor)
        activationTensor = self.decodingLinear2(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.positionTensorShape[0], \
                                                 self.positionTensorShape[1], self.positionTensorShape[2], \
                                                 self.positionTensorShape[3])
        return activationTensor  # No Sigmoid: If an entry is negative in this activation tensor, the corresponding
        # entry in the high level tensor will be a 0. Otherwise, it will be a 1. Cf. ConvertToOneHotPositionTensor(*)


    def Save(self, directory, filenameSuffix):
        filename = 'AutoencoderNet_' + str(self.positionTensorShape) + '_' + \
                                str(self.bodyStructureList) + '_' + str(self.numberOfLatentVariables) + '_' + \
                                ('zeroPadding_' if self.zeroPadding else 'noZeroPadding_') + \
                                filenameSuffix + '.pth'
        # Remove spaces
        filename = filename.replace(' ', '')
        filepath = os.path.join(directory, filename)
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath):
        filename = os.path.basename(filepath)
        # Tokenize the filename with '_'
        tokens = filename.split('_')
        if len(tokens) < 6:
            raise ValueError("Net.Load(): The number of tokens of {} is less than 6 ({})".format(filename, len(tokens)))
        positionTensorShape = ast.literal_eval(tokens[1])
        bodyStructureList = ast.literal_eval(tokens[2])
        numberOfLatentVariables = int(tokens[3])
        #print("Net.Load(): tokens[4] = {}".format(tokens[4]))
        zeroPadding = False if tokens[4] == 'noZeroPadding' else True
        #print ("Net.Load(): zeroPadding = {}".format(zeroPadding))
        self.__init__(positionTensorShape, bodyStructureList, numberOfLatentVariables, zeroPadding)
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))
        self.eval()

    def EncodingLayers(self):
        return self.bodyStructureSeq, self.fullyConnectedLayer

    def Shapes(self):
        return self.positionTensorShape, self.bodyActivationNumberOfEntries, self.numberOfLatentVariables, self.bodyStructureList

    """def AddConvolutionLayer(self, kernelDimension):
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
    """

def ConvolutionLayer(inputNumberOfChannels, kernelDimension, numberOfOutputChannels, stride=1, dilation=1, zeroPadding=True):
    if zeroPadding:
        padding = (int(kernelDimension[0]/2), int(kernelDimension[1]/2), int(kernelDimension[2]/2) )
    else:
        padding = 0
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels=inputNumberOfChannels,
                        out_channels=numberOfOutputChannels,
                        kernel_size=kernelDimension,
                        padding=padding,
                        stride=stride,
                        dilation=dilation),
        torch.nn.ReLU())

def ConvertToOneHotPositionTensor(reconstructedPositionTensor): # Convert to a high level position tensor, with 0 and 1's
    positionTensorShape = reconstructedPositionTensor.shape #([N,] C, D, H, W)
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
                        highestValue = -1.0E9
                        highestChannelNdx = -1
                        for channelNdx in range(positionTensorShape[1]):
                            value = reconstructedPositionTensor[minibatchNdx][channelNdx][depth][row][column]
                            if value > highestValue:
                                highestValue = value
                                highestChannelNdx = channelNdx
                        if highestValue >= 0.0:
                            oneHotPositionTensor[minibatchNdx][highestChannelNdx][depth][row][column] = 1
    else:
        for depth in range(positionTensorShape[1]):
            for row in range(positionTensorShape[2]):
                for column in range(positionTensorShape[3]):
                    highestValue = -1.0E9
                    highestChannelNdx = -1
                    for channelNdx in range(positionTensorShape[0]):
                        value = reconstructedPositionTensor[channelNdx][depth][row][column]
                        if value > highestValue:
                            highestValue = value
                            highestChannelNdx = channelNdx
                    if highestValue >= 0.0:
                        oneHotPositionTensor[highestChannelNdx][depth][row][column] = 1

    return oneHotPositionTensor


def main():
    print ("autoencoder > position.py main()")

    import checkers
    authority = checkers.Authority()
    positionTensorShape = authority.PositionTensorShape()
    autoencoder = Net(positionTensorShape, [(3, 32, 2), (3, 64, 2)], 100)

    inputPositionTensor = authority.InitialPosition()
    outputTensor = autoencoder(inputPositionTensor.unsqueeze(0))

if __name__ == '__main__':
    main()