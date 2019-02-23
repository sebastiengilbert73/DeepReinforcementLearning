import argparse
import ast
import torch
import math


class NeuralNetwork(torch.nn.Module):
    def __init__(self, inputTensorSize, bodyStructure, outputTensorSize): # Both input and output tensor sizes must be (C, D, H, W)
        super(NeuralNetwork, self).__init__()
        if len(inputTensorSize) != 4:
            raise ValueError("NeuralNetwork.__init__(): The length of inputTensorSize ({}) is not 4 (C, D, H, W)".format(len(inputTensorSize)))
        if len(outputTensorSize) != 4:
            raise ValueError("NeuralNetwork.__init__(): The length of outputTensorSize ({}) is not 4 (C, D, H, W)".format(len(outputTensorSize)))
        self.inputTensorSize = inputTensorSize;

        if ast.literal_eval(bodyStructure) == [(3, 32), (3, 32)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=inputTensorSize[0], out_channels=32,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=32, out_channels=32,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.lastLayerInputNumberOfChannels = 32

        else:
            raise NotImplementedError("NeuralNetwork.__init__(): Unknown body structure '{}'".format(bodyStructure))

        self.probabilitiesChannelMatcher = torch.nn.Conv3d(in_channels=self.lastLayerInputNumberOfChannels,
                                              out_channels=outputTensorSize[0],
                                              kernel_size=1,
                                              padding=0
                                              )
        self.probabilitiesResizer = torch.nn.Upsample(size=outputTensorSize[-3:], mode='trilinear')
        self.outputTensorSize = outputTensorSize
        self.lastLayerNumberOfFeatures = self.lastLayerInputNumberOfChannels * \
            outputTensorSize[-3] * outputTensorSize[-2] * outputTensorSize [-1]
        self.valueHead = torch.nn.Sequential(
            torch.nn.Linear(self.lastLayerNumberOfFeatures, math.ceil(math.sqrt(self.lastLayerNumberOfFeatures))),
            torch.nn.ReLU(),
            torch.nn.Linear(math.ceil(math.sqrt(self.lastLayerNumberOfFeatures)), 1)
        )

    def forward(self, inputs):
        # Compute the output of the body
        bodyOutputTensor = self.bodyStructure(inputs)

        # Move probabilities
        moveProbabilitiesActivation = self.probabilitiesChannelMatcher(bodyOutputTensor)
        #print ("NeuralNetwork.forward(): activation.shape = {}".format(activation.shape))
        moveProbabilitiesTensor = self.probabilitiesResizer(moveProbabilitiesActivation)

        # Value
        #print ("NeuralNetwork.forward(): bodyOutputTensor.shape = {}".format(bodyOutputTensor.shape))
        bodyOutputVector = bodyOutputTensor.view(-1, self.lastLayerNumberOfFeatures)
        valueActivation = self.valueHead(bodyOutputVector)
        #print ("NeuralNetwork.forward(): valueActivation = {}".format(valueActivation))
        return moveProbabilitiesTensor, valueActivation


def NormalizeProbabilities(moveProbabilitiesTensor, legalMovesMask, preApplySoftMax=True, softMaxTemperature=1.0):
    if moveProbabilitiesTensor.shape != legalMovesMask.shape:
        raise ValueError("NormalizeProbabilities(): The shape of moveProbabilitiesTensor ({}) doesn't match the shape of legalMovesMask ({})".format(moveProbabilitiesTensor, legalMovesMask))

    # Flatten the tensors
    moveProbabilitiesVector = moveProbabilitiesTensor.view(moveProbabilitiesTensor.numel())
    legalMovesVector = legalMovesMask.view(legalMovesMask.numel())
    for index in range(moveProbabilitiesVector.shape[0]):
        if legalMovesVector[index] == 0:
            moveProbabilitiesVector[index] = 0
    if preApplySoftMax:
        moveProbabilitiesVector = torch.nn.functional.softmax(moveProbabilitiesVector/softMaxTemperature)
    else: # Normalize
        sum = 0
        for index in range(moveProbabilitiesVector.shape[0]):
            if moveProbabilitiesVector[index] < 0:
                raise ValueError("NormalizeProbabilities(): The probability value {} is negative".format(moveProbabilitiesVector[index]))
            sum += moveProbabilitiesVector[index]
        moveProbabilitiesVector = moveProbabilitiesVector/sum

    # Resize to original size
    normalizedProbabilitiesTensor = moveProbabilitiesVector.view(moveProbabilitiesTensor.shape)
    return normalizedProbabilitiesTensor

def main():
    print ("policy.py main()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodyStructure', help="The structure of the neural network body. Default: '[(3, 32), (3, 32)]'", default='[(3, 32), (3, 32)]')
    args = parser.parse_args()

    inputTensorSize = (2, 1, 3, 3) # Tic-tac-toe positions (C, D, H, W)
    outputTensorSize = (1, 1, 3, 3) # Tic-tac-toe moves (C, D, H, W)
    neuralNet = NeuralNetwork(inputTensorSize, args.bodyStructure, outputTensorSize)
    input = torch.zeros(inputTensorSize).unsqueeze(0) # Add a dummy mini-batch
    input[0, 0, 0, 0, 0] = 1.0
    output = neuralNet(input)
    print ("main(): output = {}".format(output))
    print ("main(): output[0].shape = {}".format(output[0].shape))


if __name__ == '__main__':
    main()