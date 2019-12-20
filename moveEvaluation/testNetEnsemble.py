import torch
import utilities
import expectedMoveValues
import generateMoveStatistics
import ConvolutionStack
import statistics
import netEnsemble


def main():
    print ("testNetEnsemble.py main()")
    neuralNet2 = ConvolutionStack.Net()
    neuralNet2.Load(
        '/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033.pth')
    neuralNet3 = ConvolutionStack.Net()
    neuralNet3.Load(
        '/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033b.pth')
    neuralNet4 = ConvolutionStack.Net()
    neuralNet4.Load(
        '/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033c.pth')
    neuralNet5 = ConvolutionStack.Net()
    neuralNet5.Load(
        '/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033d.pth')
    neuralNet6 = ConvolutionStack.Net()
    neuralNet6.Load(
        '/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.0033e.pth')
    neuralNet7 = ConvolutionStack.Net()
    neuralNet7.Load(
        '/home/sebastien/projects/DeepReinforcementLearning/outputs/ToKeep/Net_(2,1,6,7)_[(5,32),(5,32),(5,32)]_(1,1,1,7)_connect4_defeatRate0.pth')
    neuralNetsList = [neuralNet2, neuralNet3, neuralNet4, neuralNet5, neuralNet6, neuralNet7]

    committee = netEnsemble.Committee(neuralNetsList)

    import connect4
    authority = connect4.Authority()
    playerList = authority.PlayersList()
    epsilon = 0
    maximumDepthOfSemiExhaustiveSearch = 2
    numberOfTopMovesToDevelop = 7
    inputTensor = authority.InitialPosition()
    inputTensor[0, 0, 5, 1] = 1
    inputTensor[0, 0, 2, 2] = 1
    inputTensor[0, 0, 4, 2] = 1
    inputTensor[0, 0, 5, 3] = 1
    inputTensor[0, 0, 5, 4] = 1
    inputTensor[0, 0, 5, 5] = 1
    inputTensor[1, 0, 3, 2] = 1
    inputTensor[1, 0, 5, 2] = 1
    inputTensor[1, 0, 3, 3] = 1
    inputTensor[1, 0, 4, 3] = 1
    inputTensor[1, 0, 3, 4] = 1
    inputTensor[1, 0, 4, 4] = 1
    inputTensor[1, 0, 4, 6] = 1
    inputTensor[1, 0, 5, 6] = 1
    authority.Display(inputTensor)
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