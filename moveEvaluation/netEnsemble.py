import torch
import utilities
import expectedMoveValues
import generateMoveStatistics
import ConvolutionStack
import statistics


class Committee():
    def __init__(self, neuralNetworksList):
        self.members = neuralNetworksList

    def forward(self, inputTensor):
        outputTensorsList = []
        for member in self.members:
            outputTensor = member.forward(inputTensor)
            outputTensorsList.append(outputTensor)
        return outputTensorsList

    def MedianValues(self, inputTensor):
        outputTensorsList = self.forward(inputTensor)
        medianValuesTensorShape = outputTensorsList[0].shape
        medianValuesTensor = torch.zeros(medianValuesTensorShape)

        for sampleNdx in range(medianValuesTensorShape[0]):
            for channelNdx in range(medianValuesTensorShape[1]):
                for depthNdx in range(medianValuesTensorShape[2]):
                    for heightNdx in range(medianValuesTensorShape[3]):
                        for widthNdx in range(medianValuesTensorShape[4]):
                            valuesList = []
                            for memberNdx in range(len(outputTensorsList)):
                                valuesList.append(outputTensorsList[memberNdx][sampleNdx, channelNdx, depthNdx, heightNdx, widthNdx])
                            medianValue = statistics.median(valuesList)
                            medianValuesTensor[sampleNdx, channelNdx, depthNdx, heightNdx, widthNdx] = medianValue

        return medianValuesTensor

    def __call__(self, inputTensor): # To implement outputTensor = committee(inputTensor), as an in-place equivalent to outputTensor = neuralNetwork(inputTensor)
        return self.MedianValues(inputTensor)

def main():
    print ('netEnsemble.py main()')
    neuralNet1 = ConvolutionStack.Net()
    neuralNet1.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,16),(5,16),(5,16)]_(1,1,1,7)_connect4_356.pth')
    neuralNet2 = ConvolutionStack.Net()
    neuralNet2.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033.pth')
    neuralNet3 = ConvolutionStack.Net()
    neuralNet3.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033b.pth')
    neuralNet4 = ConvolutionStack.Net()
    neuralNet4.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033c.pth')
    neuralNet5 = ConvolutionStack.Net()
    neuralNet5.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033d.pth')
    neuralNet6 = ConvolutionStack.Net()
    neuralNet6.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033e.pth')
    neuralNet7 = ConvolutionStack.Net()
    neuralNet7.Load('/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.pth')
    neuralNetsList = [neuralNet1, neuralNet2, neuralNet3, neuralNet4, neuralNet5, neuralNet6, neuralNet7]

    committee = Committee(neuralNetsList)

    inputTensor = torch.zeros((2, 1, 6, 7)).unsqueeze(0)
    outputTensorsList = committee.forward(inputTensor)
    print ('main(): outputTensorsList = {}'.format(outputTensorsList))

    medianValuesTensor = committee.MedianValues(torch.zeros((2, 1, 6, 7)).unsqueeze(0))
    print ("medianValuesTensor = {}".format(medianValuesTensor))

    import connect4
    authority = connect4.Authority()
    playerList = authority.PlayersList()
    epsilon = 0
    maximumDepthOfSemiExhaustiveSearch = 2
    numberOfTopMovesToDevelop = 3
    inputTensor = authority.InitialPosition()
    (moveValuesTensor, standardDeviationTensor, legalMovesMask) = expectedMoveValues.SemiExhaustiveMiniMax(
        playerList,
        authority,
        committee,
        inputTensor,
        epsilon,
        maximumDepthOfSemiExhaustiveSearch,
        1,
        numberOfTopMovesToDevelop
    )
    print ("moveValuesTensor = \n{}".format(moveValuesTensor))
    print ("standardDeviationTensor = \n{}".format(standardDeviationTensor))
    print ("legalMovesMask = \n{}".format(legalMovesMask))

if __name__ == '__main__':
    main()