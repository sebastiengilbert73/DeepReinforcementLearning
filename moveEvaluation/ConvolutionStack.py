import ast
import torch
import collections # OrderedDict

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
        linearActivation = torch.nn.functional.relu(self.fullyConnectedLayer(bodyActivationVector))
        #print ("Net.forward(): linearActivation.shape = {}".format(linearActivation.shape))
        outputTensor = linearActivation.view((minibatchSize,
                                              self.outputTensorSize[0],
                                              self.outputTensorSize[1],
                                              self.outputTensorSize[2],
                                              self.outputTensorSize[3],
                                              ))
        #channelsMatcherActivation = self.actionValuesChannelMatcher(bodyActivation)
        return outputTensor

def main():
    print ("ConvolutionStack.py main()")
    inputTensorSize = (2, 1, 3, 4)
    outputTensorSize = (5, 1, 4, 7)
    neuralNet = Net(inputTensorSize, [(3, 32), (5, 16), (7, 8)], outputTensorSize)
    inputTensor = torch.randn((27, 2, 1, 3, 4))
    outputTensor = neuralNet(inputTensor)
    print ("main(): outputTensor.shape = {}".format(outputTensor.shape))

if __name__ == '__main__':
    main()