import ast
import torch
import collections # OrderedDict
import expectedMoveValues
import generateMoveStatistics
import utilities
import random
import numpy
import os
import autoencoder.position # Position autoencoders

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

    """def ChooseAMove(self, positionTensor, player, gameAuthority, chooseHighestProbabilityIfAtLeast,
                    preApplySoftMax, softMaxTemperature,
                    epsilon):
        actionValuesTensor = self.forward(positionTensor.unsqueeze(0)) # Add a dummy minibatch
        # Remove the dummy minibatch
        actionValuesTensor = torch.squeeze(actionValuesTensor, 0)
        #print ("Net.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))

        chooseARandomMove = random.random() < epsilon
        if chooseARandomMove:
            #print ("Net.ChooseAMove(): Choosing a random move")
            return utilities.ChooseARandomMove(positionTensor, player, gameAuthority)

        # Else: choose according to probabilities
        legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
        if torch.nonzero(legalMovesMask).size(0) == 0:
            return None

        normalizedActionValuesTensor = utilities.NormalizeProbabilities(actionValuesTensor,
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
    """

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
        self.eval()

    def HighestActionValueMove(self, positionTensor, player, authority):
        return self.ChooseAMove(
            positionTensor,
            player,
            authority,
            0.0,
            True,
            1.0,
            0.0
            )

"""def ChooseAMove(neuralNetwork, positionTensor, player, gameAuthority, chooseHighestProbabilityIfAtLeast,
                preApplySoftMax, softMaxTemperature,
                epsilon):
    actionValuesTensor = neuralNetwork(positionTensor.unsqueeze(0)) # Add a dummy minibatch
    # Remove the dummy minibatch
    actionValuesTensor = torch.squeeze(actionValuesTensor, 0)
    #print ("Net.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))

    chooseARandomMove = random.random() < epsilon
    if chooseARandomMove:
        #print ("Net.ChooseAMove(): Choosing a random move")
        return utilities.ChooseARandomMove(positionTensor, player, gameAuthority)

    # Else: choose according to probabilities
    legalMovesMask = gameAuthority.LegalMovesMask(positionTensor, player)
    if torch.nonzero(legalMovesMask).size(0) == 0:
        return None

    normalizedActionValuesTensor = utilities.NormalizeProbabilities(actionValuesTensor,
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
        highestProbabilityCoords = utilities.CoordinatesFromFlatIndex(
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
        print ("ConvolutionStack.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
        print ("ConvolutionStack.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
        print ("ConvolutionStack.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
        print ("ConvolutionStack.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
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

    
    if chosenCoordinates is None:
        print ("ConvolutionStack.ChooseAMove(): positionTensor = \n{}".format(positionTensor))
        print ("ConvolutionStack.ChooseAMove(): actionValuesTensor = \n{}".format(actionValuesTensor))
        print ("ConvolutionStack.ChooseAMove(): legalMovesMask =\n{}".format(legalMovesMask))
        print ("ConvolutionStack.ChooseAMove(): normalizedActionValuesTensor =\n{}".format(normalizedActionValuesTensor))
        print ("ConvolutionStack.ChooseAMove(): runningSum = {}; randomNbr = {}".format(runningSum, randomNbr))
        raise IndexError("ConvolutionStack.ChooseAMove(): chosenCoordinates is None...!???")

    chosenMoveArr = numpy.zeros(gameAuthority.MoveTensorShape())
    chosenMoveArr[chosenCoordinates] = 1.0
    return torch.from_numpy(chosenMoveArr).float()
"""

"""def CoordinatesFromFlatIndex(
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
"""

class ActionValueDecoder(torch.nn.Module):
    def __init__(self,
                 inputTensorShape=(6, 1, 8, 8),
                 encodingBodyStructureList=[(3, 16, 2)],
                 encodingBodyStructureSeq=None,              # Obtained from an autoencoder
                 encodingBodyActivationNumberOfEntries=64, # Obtained from an autoencoder
                 encodingFullyConnectedLayer=None,                   # Obtained from an autoencoder
                 numberOfLatentVariables=100,               # Obtained from an autoencoder
                 decodingIntermediaryLayerShapes=[(16, 1, 2, 2), (8, 1, 4, 4)],
                 outputTensorShape=(4, 1, 8, 8)):  # Both input and output tensor shapes must be (C, D, H, W)
        super(ActionValueDecoder, self).__init__()
        self.inputTensorShape = inputTensorShape
        self.encodingBodyStructureList = encodingBodyStructureList
        self.layerNameToTensorPhysicalShapeDict = {}
        previousLayerPhysicalShape = (
            self.inputTensorShape[1], self.inputTensorShape[2], self.inputTensorShape[3])
        numberOfInputChannels = inputTensorShape[0]
        bodyStructureDict = collections.OrderedDict()
        self.encodingLayerNames = []
        if encodingBodyStructureSeq is None:
            for layerNdx in range(len(encodingBodyStructureList)):
                kernelDimensionNumberTrio = encodingBodyStructureList[layerNdx]
                if len(kernelDimensionNumberTrio) != 3:
                    raise ValueError(
                        "ActionValueDecoder.__init__(): The length of the tuple {} is not 3".format(kernelDimensionNumberTrio))
                layerName = 'layer_' + str(layerNdx)
                stride = kernelDimensionNumberTrio[2]

                self.layerNameToTensorPhysicalShapeDict[layerName] = (
                max((previousLayerPhysicalShape[0] + 1) // stride, 1),
                max((previousLayerPhysicalShape[1] + 1) // stride, 1),
                max((previousLayerPhysicalShape[2] + 1) // stride, 1))

                previousLayerPhysicalShape = self.layerNameToTensorPhysicalShapeDict[layerName]
                bodyStructureDict[layerName] = autoencoder.position.ConvolutionLayer(numberOfInputChannels,
                                                                     kernelDimensionNumberTrio[0],
                                                                     kernelDimensionNumberTrio[1],
                                                                     stride,
                                                                     dilation=1)
                self.encodingLayerNames.append(layerName)
                numberOfInputChannels = kernelDimensionNumberTrio[1]
            self.encodingBodyStructureSeq = torch.nn.Sequential(bodyStructureDict)
        else:
            self.encodingBodyStructureSeq = encodingBodyStructureSeq

        self.encodingBodyActivationNumberOfEntries = encodingBodyActivationNumberOfEntries
        if encodingFullyConnectedLayer is None:
            self.encodingFullyConnectedLayer = torch.nn.Linear(self.encodingBodyActivationNumberOfEntries, numberOfLatentVariables)
        else:
            self.encodingFullyConnectedLayer = encodingFullyConnectedLayer

        self.numberOfLatentVariables = numberOfLatentVariables
        self.decodingIntermediaryLayerShapes = decodingIntermediaryLayerShapes
        self.outputTensorShape = outputTensorShape

        self.decodingInputNumberOfEntries = self.decodingIntermediaryLayerShapes[0][0] * \
                                            self.decodingIntermediaryLayerShapes[0][1] * \
                                            self.decodingIntermediaryLayerShapes[0][2] * \
                                            self.decodingIntermediaryLayerShapes[0][3]
        self.decodingFullyConnectedLayer = torch.nn.Linear(self.numberOfLatentVariables, self.decodingInputNumberOfEntries)

        # Transpose convolutions
        decodingTransposeConvolutionsDict = collections.OrderedDict()
        numberOfTransposeConvolutions = len(self.decodingIntermediaryLayerShapes)
        for transposeConvolutionlayerNdx in range(numberOfTransposeConvolutions):
            layerName = 'decodingLayer_' + str(transposeConvolutionlayerNdx)
            originalShape = self.decodingIntermediaryLayerShapes[transposeConvolutionlayerNdx]
            if transposeConvolutionlayerNdx == len(self.decodingIntermediaryLayerShapes) - 1:
                finalShape = self.outputTensorShape
            else:
                finalShape = self.decodingIntermediaryLayerShapes[transposeConvolutionlayerNdx + 1]
            #print ("__init__(): originalShape = {}; finalShape = {}".format(originalShape, finalShape))
            numberOfInputChannels = originalShape[0]
            numberOfOutputChannels = finalShape[0]
            kernel_size = 3
            stride = 1
            padding = 1

            #print ("__init__(): originalShape = {}; finalShape = {}".format(originalShape, finalShape))
            dilation = (finalShape[1] // originalShape[1], finalShape[2] // originalShape[2], finalShape[3] // originalShape[3])
            output_padding = (dilation[0] - 1, dilation[1] - 1, dilation[2] - 1 )
            decodingTransposeConvolutionsDict[layerName] = \
                torch.nn.ConvTranspose3d(
                    in_channels=numberOfInputChannels,
                    out_channels=numberOfOutputChannels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=1,
                    bias=True,
                    dilation=dilation
                )
        self.decodingTransposeConvolutionsSeq = torch.nn.Sequential(decodingTransposeConvolutionsDict)
        self.ReLU = torch.nn.ReLU()
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, inputTensor):
        minibatchSize = inputTensor.shape[0]

        activationTensor = inputTensor
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        activationTensor = self.ReLU(activationTensor)

        # Decoding
        activationTensor = self.decodingFullyConnectedLayer(activationTensor)
        activationTensor = self.ReLU(activationTensor)
        #print ("forward(): self.decodingIntermediaryLayerShapes[0] = {}".format(self.decodingIntermediaryLayerShapes[0]))
        activationTensor = activationTensor.view(minibatchSize, self.decodingIntermediaryLayerShapes[0][0], \
                                                 self.decodingIntermediaryLayerShapes[0][1], \
                                                 self.decodingIntermediaryLayerShapes[0][2], \
                                                 self.decodingIntermediaryLayerShapes[0][3])
        activationTensor = self.decodingTransposeConvolutionsSeq(activationTensor)
        """activationTensor = activationTensor.view(minibatchSize, self.outputTensorShape[0], \
                                                 self.outputTensorShape[1], self.outputTensorShape[2], \
                                                 self.outputTensorShape[3])
        """
        activationTensor = -1.0 + 2.0 * self.Sigmoid(activationTensor)
        return activationTensor

    def Save(self, directory, filenameSuffix):
        filename = 'ActionValueDecoder_' + str(self.inputTensorShape) + '_' + \
                   str(self.encodingBodyStructureList) + '_' + \
                   str(self.encodingBodyActivationNumberOfEntries) + '_' + \
                   str(self.numberOfLatentVariables) + '_' + \
                   str(self.decodingIntermediaryLayerShapes) + '_' + \
                   str(self.outputTensorShape) + '_' + \
                   filenameSuffix + '.pth'
        # Remove spaces
        filename = filename.replace(' ', '')
        filepath = os.path.join(directory, filename)
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath):
        filename = os.path.basename(filepath)

        # Tokenize the filename with '_'
        tokens = filename.split('_')
        if len(tokens) < 7:
            raise ValueError("ActionValueDecoder.Load(): The number of tokens of {} is less than 7 ({})".format(filename, len(tokens)))
        inputTensorShape = ast.literal_eval(tokens[1])
        encodingBodyStructureList = ast.literal_eval(tokens[2])
        encodingBodyActivationNumberOfEntries = int(tokens[3])
        numberOfLatentVariables = int(tokens[4])
        decodingIntermediaryLayerShapes = ast.literal_eval(tokens[5])
        outputTensorShape = ast.literal_eval(tokens[6])
        self.__init__(
            inputTensorShape,
            encodingBodyStructureList,
            None,
            encodingBodyActivationNumberOfEntries,
            None,
            numberOfLatentVariables,
            decodingIntermediaryLayerShapes,
            outputTensorShape)
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))
        self.eval()

def BuildAnActionValueDecoderFromAnAutoencoder(autoencoderNet, intermediateTensorShapesList, outputTensorShape):
    encodingBodyStructureSeq, encodingFullyConnectedLayer = autoencoderNet.EncodingLayers()
    positionTensorShape, bodyActivationNumberOfEntries, numberOfLatentVariables, bodyStructureList = autoencoderNet.Shapes()
    actionValueDecoder = ActionValueDecoder(
        inputTensorShape=positionTensorShape,
        encodingBodyStructureList=bodyStructureList,
        encodingBodyStructureSeq=encodingBodyStructureSeq,  # Obtained from an autoencoder
        encodingBodyActivationNumberOfEntries=bodyActivationNumberOfEntries,  # Obtained from an autoencoder
        encodingFullyConnectedLayer=encodingFullyConnectedLayer,  # Obtained from an autoencoder
        numberOfLatentVariables=numberOfLatentVariables,  # Obtained from an autoencoder
        decodingIntermediaryLayerShapes=intermediateTensorShapesList,
        outputTensorShape=outputTensorShape)
    return actionValueDecoder

def main():

    print ("ConvolutionStack.py main()")
    import autoencoder.position
    import checkers



    autoencoderNet = autoencoder.position.Net()
    autoencoderNet.Load('/home/sebastien/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(6,1,8,8)_[(5,16,2),(5,32,2)]_200_checkersAutoencoder_44.pth')

    actionValueDecoder = BuildAnActionValueDecoderFromAnAutoencoder(autoencoderNet, [(24, 1, 2, 2), (12, 1, 4, 4)], (4, 1, 8, 8))

    authority = checkers.Authority()
    inputTensor = authority.InitialPosition()
    outputTensor = actionValueDecoder(inputTensor.unsqueeze(0))
    print ("outputTensor = {}".format(outputTensor))

    actionValueDecoder.Save('./', 'test2')

    #actionValueDecoder = ActionValueDecoder()
    #actionValueDecoder.Load('/home/sebastien/projects/DeepReinforcementLearning/moveEvaluation/ActionValueDecoder_(6,1,8,8)_[(5,16,2),(5,32,2)]_128_200_[(16,1,2,2),(8,1,4,4)]_(4,1,8,8)_test1.pth')

if __name__ == '__main__':
    main()