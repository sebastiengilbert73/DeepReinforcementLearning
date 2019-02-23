import argparse
import ast
import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, inputTensorSize, bodyStructure, outputTensorSize): # Both input and output tensor sizes must be (N, C, H, W)
        super(NeuralNetwork, self).__init__()
        if len(inputTensorSize) != 4:
            raise ValueError("NeuralNetwork.__init__(): The length of inputTensorSize ({}) is not 4".format(len(inputTensorSize)))
        if len(outputTensorSize) != 4:
            raise ValueError("NeuralNetwork.__init__(): The length of outputTensorSize ({}) is not 4".format(len(outputTensorSize)))
        self.inputTensorSize = inputTensorSize;

        if ast.literal_eval(bodyStructure) == [(3, 32), (3, 32)]:
            self.bodyStructure = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inputTensorSize[1], out_channels=32,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=32, out_channels=32,
                                kernel_size=3,
                                padding=1),
                torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True,
                                     track_running_stats=True),
                torch.nn.ReLU()
            )
            self.channelMatcher = torch.nn.Conv2d(in_channels=32, out_channels=outputTensorSize[1],
                                                  kernel_size=1,
                                                  padding=0
                                                  )
        else:
            raise NotImplementedError("NeuralNetwork.__init__(): Unknown body structure '{}'".format(bodyStructure))


        self.resizer = torch.nn.Upsample(size=outputTensorSize[-2:], mode='bilinear')
        self.outputTensorSize = outputTensorSize

    def forward(self, inputs):
        activation = self.bodyStructure(inputs)
        activation = self.channelMatcher(activation)
        print ("NeuralNetwork.forward(): activation.shape = {}".format(activation.shape))
        output = self.resizer(activation)
        return output


def main():
    print ("policy.py main()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodyStructure', help="The structure of the neural network body. Default: '[(3, 32), (3, 32)]'", default='[(3, 32), (3, 32)]')
    args = parser.parse_args()

    inputTensorSize = (1,2,3,3) # Tic-tac-toe
    outputTensorSize = (1, 1, 3, 3)
    neuralNet = NeuralNetwork(inputTensorSize, args.bodyStructure, outputTensorSize)
    input = torch.zeros(inputTensorSize)
    input[0, 0, 0, 0] = 1.0
    output = neuralNet(input)
    print ("main(): output = {}".format(output))
    print ("main(): output.shape = {}".format(output.shape))

if __name__ == '__main__':
    main()