import Comparison
import autoencoder.position
import torch
import math
import numpy

class DecoderClassifier(Comparison.Comparator,
                        torch.nn.Module):
    def __init__(self,
                 inputTensorShape=(2, 1, 3, 3),
                 encodingBodyStructureList=[(3, 16, 1)],
                 encodingBodyStructureSeq=None,  # Obtained from an autoencoder
                 encodingBodyActivationNumberOfEntries=64,  # Obtained from an autoencoder
                 encodingFullyConnectedLayer=None,  # Obtained from an autoencoder
                 numberOfLatentVariables=100,  # Obtained from an autoencoder
                 dropoutRatio=0.1,
                 ):
        super(DecoderClassifier, self).__init__()
        self.inputTensorShape = inputTensorShape
        self.encodingBodyStructureList = encodingBodyStructureList
        self.encodingBodyStructureSeq = encodingBodyStructureSeq
        self.encodingBodyActivationNumberOfEntries = encodingBodyActivationNumberOfEntries
        self.encodingFullyConnectedLayer = encodingFullyConnectedLayer
        self.numberOfLatentVariables = numberOfLatentVariables

        self.decodingIntermediateNumberOfNeurons = 3 * numberOfLatentVariables #math.ceil(math.sqrt(2 * numberOfLatentVariables * 2))
        self.decodingLinearLayer1 = torch.nn.Linear(2 * self.numberOfLatentVariables, self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer2 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons,
                                                    self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer3 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons,
                                                    self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer4 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons,
                                                    self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer5 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons,
                                                    self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer6 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons,
                                                    self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer7 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons,
                                                    self.decodingIntermediateNumberOfNeurons)
        self.decodingLinearLayer8 = torch.nn.Linear(self.decodingIntermediateNumberOfNeurons, 2)

        self.batchnorm = torch.nn.BatchNorm1d(num_features = self.decodingIntermediateNumberOfNeurons)
        self.dropout = torch.nn.Dropout(p=dropoutRatio)

    def LatentVariables(self, positionBatch):
        minibatchSize = positionBatch.shape[0]
        activationTensor = positionBatch
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        latentVariablesTensor = torch.nn.functional.relu(activationTensor)

        return latentVariablesTensor

    def ConcatenatedLatentVariables(self, inputTensor):
        if inputTensor.shape[1] != 2 * self.inputTensorShape[0]:
            raise ValueError("ConcatenatedLatentVariables(): The input tensor shape number of channels ({}) is not 2 * self.inputTensorShape[0] ({})".format(inputTensor.shape[1], 2 * self.inputTensorShape[0]))
        latentVariablesTsr0 = self.LatentVariables(inputTensor[:, 0: self.inputTensorShape[0], :, :, :])
        latentVariablesTsr1 = self.LatentVariables(inputTensor[:, self.inputTensorShape[0]: , :, :, :])
        concatenatedLatentVariablesTsr = torch.cat([latentVariablesTsr0, latentVariablesTsr1], dim=1)
        return concatenatedLatentVariablesTsr

    def forward(self, inputTensor):
        """if inputTensor.shape[1] != 2 * self.inputTensorShape[0]:
            raise ValueError("forward(): The input tensor shape number of channels ({}) is not 2 * self.inputTensorShape[0] ({})".format(inputTensor.shape[1], 2 * self.inputTensorShape[0]))
        latentVariablesTsr0 = self.LatentVariables(inputTensor[:, 0: self.inputTensorShape[0], :, :, :])
        latentVariablesTsr1 = self.LatentVariables(inputTensor[:, self.inputTensorShape[0]: , :, :, :])
        latentVariablesTsr = torch.cat([latentVariablesTsr0, latentVariablesTsr1], dim=1)
        """
        latentVariablesTsr = self.ConcatenatedLatentVariables(inputTensor)
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer1(latentVariablesTsr))
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer2(activationTsr))
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer3(activationTsr))
        activationTsr = self.dropout(activationTsr)
        activationTsr = self.batchnorm(activationTsr)
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer4(activationTsr))
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer5(activationTsr))
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer6(activationTsr))
        activationTsr = self.dropout(activationTsr)
        activationTsr = self.batchnorm(activationTsr)
        activationTsr = torch.nn.functional.relu(self.decodingLinearLayer7(activationTsr))
        outputTsr = torch.sigmoid( self.decodingLinearLayer8(activationTsr) )
        return outputTsr

    def BestPosition(self, position0, position1):
        inputTsr = torch.zeros(1, 2 * self.inputTensorShape[0], self.inputTensorShape[1], self.inputTensorShape[2], self.inputTensorShape[3])
        inputTsr[0, 0: self.inputTensorShape[0], :, :, :] = position0
        inputTsr[0, self.inputTensorShape[0]: , :, :, :] = position1
        outputTsr = self.forward(inputTsr)
        if outputTsr[0, 0] > outputTsr[0, 1]:
            return position0
        elif outputTsr[0, 0] < outputTsr[0, 1]:
            return position1
        else: # Equality: return random move
            if numpy.random.random() < 0.5:
                return position0
            else:
                return position1

    def Gradient0(self):
        return self.decodingLinearLayer1.weight.grad

    def Gradient0AbsMean(self):
        gradient0 = self.Gradient0()
        return gradient0.abs().mean().item()

    def DecodingLatentRepresentation(self, decodingLayerNdx, inputTsr):
        if decodingLayerNdx == 1:
            concatenatedLatentVariablesTsr = self.ConcatenatedLatentVariables(inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer1(concatenatedLatentVariablesTsr))
            return activationTsr
        elif decodingLayerNdx == 2:
            activationTsr = self.DecodingLatentRepresentation(1, inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer2(activationTsr))
            return activationTsr
        elif decodingLayerNdx == 3:
            activationTsr = self.DecodingLatentRepresentation(2, inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer3(activationTsr))
            activationTsr = self.dropout(activationTsr)
            activationTsr = self.batchnorm(activationTsr)
            return activationTsr
        elif decodingLayerNdx == 4:
            activationTsr = self.DecodingLatentRepresentation(3, inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer4(activationTsr))
            return activationTsr
        elif decodingLayerNdx == 5:
            activationTsr = self.DecodingLatentRepresentation(4, inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer5(activationTsr))
            return activationTsr
        elif decodingLayerNdx == 6:
            activationTsr = self.DecodingLatentRepresentation(5, inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer6(activationTsr))
            activationTsr = self.dropout(activationTsr)
            activationTsr = self.batchnorm(activationTsr)
            return activationTsr
        elif decodingLayerNdx == 7:
            activationTsr = self.DecodingLatentRepresentation(6, inputTsr)
            activationTsr = torch.nn.functional.relu(self.decodingLinearLayer7(activationTsr))
            return activationTsr
        elif decodingLayerNdx == 8:
            activationTsr = self.DecodingLatentRepresentation(7, inputTsr)
            activationTsr = torch.sigmoid(self.decodingLinearLayer8(activationTsr))
            return activationTsr
        else:
            raise NotImplementedError("DecodingLatentRepresentation(): Layer {} is not implemented".format(decodingLayerNdx))

    def DecodingLayer(self, layerNdx):
        if layerNdx == 1:
            return self.decodingLinearLayer1
        elif layerNdx == 2:
            return self.decodingLinearLayer2
        elif layerNdx == 3:
            return self.decodingLinearLayer3
        elif layerNdx == 4:
            return self.decodingLinearLayer4
        elif layerNdx == 5:
            return self.decodingLinearLayer5
        elif layerNdx == 6:
            return self.decodingLinearLayer6
        elif layerNdx == 7:
            return self.decodingLinearLayer7
        elif layerNdx == 8:
            return self.decodingLinearLayer8
        else:
            raise NotImplementedError("DecodingLayer(): Did not implement layer {}".format(layerNdx))


def BuildADecoderClassifierFromAnAutoencoder(autoencoderNet, dropoutRatio):
    encodingBodyStructureSeq, encodingFullyConnectedLayer = autoencoderNet.EncodingLayers()
    positionTensorShape, bodyActivationNumberOfEntries, numberOfLatentVariables, bodyStructureList = autoencoderNet.Shapes()
    decoderClassifier = DecoderClassifier(
        inputTensorShape=positionTensorShape,
        encodingBodyStructureList=bodyStructureList,
        encodingBodyStructureSeq=encodingBodyStructureSeq,  # Obtained from an autoencoder
        encodingBodyActivationNumberOfEntries=bodyActivationNumberOfEntries,  # Obtained from an autoencoder
        encodingFullyConnectedLayer=encodingFullyConnectedLayer,  # Obtained from an autoencoder
        numberOfLatentVariables=numberOfLatentVariables,  # Obtained from an autoencoder
        dropoutRatio=dropoutRatio
        )
    return decoderClassifier


def main():
    print ("ComparisonNet.py main()")
    import tictactoe
    import utilities
    import autoencoder

    autoencoderNet = autoencoder.position.Net()
    autoencoderNet.Load('/home/sebastien/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(2,1,3,3)_[(3,64,1)]_32_noZeroPadding_tictactoeAutoencoder_1000.pth')

    decoderClassifier = BuildADecoderClassifierFromAnAutoencoder(autoencoderNet)

    authority = tictactoe.Authority()
    currentPosition = authority.InitialPosition()
    playersList = authority.PlayersList()
    candidatePositionsAfterMoveWinnerPairs = utilities.LegalCandidatePositionsAfterMove(authority, currentPosition, playersList[0])
    candidatePositionsAfterMoveList = [candidatePositionAfterMoveWinnerPair[0] for candidatePositionAfterMoveWinnerPair in candidatePositionsAfterMoveWinnerPairs]
    #print ("candidatePositionsAfterMove = {}".format(candidatePositionsAfterMove))
    tournamentWinner = Comparison.TournamentWinner(decoderClassifier, candidatePositionsAfterMoveList)
    print ("tournamentWinner = {}".format(tournamentWinner))

    positionsList, winner = Comparison.SimulateAGame(decoderClassifier, authority, )
    print (positionsList, winner)

if __name__ == '__main__':
    main()