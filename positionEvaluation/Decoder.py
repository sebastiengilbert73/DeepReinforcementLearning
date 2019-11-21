import Predictor
import autoencoder.position
import torch
from sklearn.tree import DecisionTreeRegressor
from statistics import mean

class RandomForest(Predictor.Evaluator):
    def __init__(self,
                 inputTensorShape=(2, 1, 3, 3),
                 encodingBodyStructureList=[(3, 16, 1)],
                 encodingBodyStructureSeq=None,  # Obtained from an autoencoder
                 encodingBodyActivationNumberOfEntries=64,  # Obtained from an autoencoder
                 encodingFullyConnectedLayer=None,  # Obtained from an autoencoder
                 numberOfLatentVariables=100,  # Obtained from an autoencoder
                 maximumNumberOfTrees=100,
                 treesMaximumDepth=6
        ):
        self.inputTensorShape = inputTensorShape
        self.encodingBodyStructureList = encodingBodyStructureList
        self.encodingBodyStructureSeq = encodingBodyStructureSeq
        self.encodingBodyActivationNumberOfEntries = encodingBodyActivationNumberOfEntries
        self.encodingFullyConnectedLayer = encodingFullyConnectedLayer
        self.numberOfLatentVariables = numberOfLatentVariables
        self.maximumNumberOfTrees = maximumNumberOfTrees
        self.treesMaximumDepth = treesMaximumDepth

        self.ReLU = torch.nn.ReLU()
        self.treesList = []



    def Value(self, positionBatch):
        # Compute latent variables
        minibatchSize = positionBatch.shape[0]

        activationTensor = positionBatch
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        latentVariablesTensor = self.ReLU(activationTensor)
        #print ("RandomForest.Value(): latentVariablesTensor = \n{}".format(latentVariablesTensor))
        #print("RandomForest.Value(): latentVariablesTensor.shape = {}".format(latentVariablesTensor.shape))

        treeValuesList = []
        for exampleNdx in range(minibatchSize):
            if len(self.treesList) == 0: # There is no tree: return a dummy value for each example
                treeValuesList.append(0.0)
            else:
                featuresTensor = latentVariablesTensor[exampleNdx]
                predictionsList = []
                for treeNdx in range(len(self.treesList)):
                    #print ("Value(): featuresTensor.shape = {}".format(featuresTensor.shape))
                    #print("Value(): featuresTensor.detach().shape = {}".format(featuresTensor.detach().shape))
                    prediction = self.treesList[treeNdx].predict(featuresTensor.detach().numpy().reshape(1, -1))[0]
                    #print ("Value(): prediction = {}".format(prediction))
                    predictionsList.append(prediction)
                #print ("Value(): predictionsList = {}".format(predictionsList))
                average = mean(predictionsList)
                treeValuesList.append(average)
        return treeValuesList

    def LearnFromMinibatch(self, minibatchFeaturesTensor, minibatchTargetValues):
        # Compute latent variables
        minibatchSize = minibatchFeaturesTensor.shape[0]

        activationTensor = minibatchFeaturesTensor
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        latentVariablesTensor = self.ReLU(activationTensor)

        while len(self.treesList) >= self.maximumNumberOfTrees: # Remove a new tree
            treeErrorSumsList = []
            for treeNdx in range(len(self.treesList)):
                treeErrorSum = 0
                for exampleNdx in range(minibatchSize):
                    featuresTensor = latentVariablesTensor[exampleNdx]
                    prediction = self.treesList[treeNdx].predict(featuresTensor.numpy())
                    squaredError = (prediction - minibatchTargetValues[exampleNdx].item())**2
                    treeErrorSum += squaredError
                treeErrorSumsList.append(treeErrorSum)
            # Find the tree with the highest error
            highestError = -1
            worstTreeNdx = -1
            for treeNdx in range(len(treeErrorSumsList)):
                if treeErrorSumsList[treeNdx] > highestError:
                    highestError = treeErrorSumsList[treeNdx]
                    worstTreeNdx = treeNdx
            # Remove the worst tree
            del(self.treesList[worstTreeNdx])

        tree = DecisionTreeRegressor()
        tree.fit(latentVariablesTensor.detach().numpy(), minibatchTargetValues.detach().numpy())
        self.treesList.append(tree)




def BuildARandomForestDecoderFromAnAutoencoder(autoencoderNet, maximumNumberOfTrees, treesMaximumDepth):
    encodingBodyStructureSeq, encodingFullyConnectedLayer = autoencoderNet.EncodingLayers()
    positionTensorShape, bodyActivationNumberOfEntries, numberOfLatentVariables, bodyStructureList = autoencoderNet.Shapes()
    randomForestDecoder = RandomForest(
        inputTensorShape=positionTensorShape,
        encodingBodyStructureList=bodyStructureList,
        encodingBodyStructureSeq=encodingBodyStructureSeq,  # Obtained from an autoencoder
        encodingBodyActivationNumberOfEntries=bodyActivationNumberOfEntries,  # Obtained from an autoencoder
        encodingFullyConnectedLayer=encodingFullyConnectedLayer,  # Obtained from an autoencoder
        numberOfLatentVariables=numberOfLatentVariables,  # Obtained from an autoencoder
        maximumNumberOfTrees=maximumNumberOfTrees,
        treesMaximumDepth=treesMaximumDepth)
    return randomForestDecoder

def main():
    print("Decoder.py main()")
    autoencoderNN = autoencoder.position.Net()
    autoencoderNN.Load('/home/segilber/projects/DeepReinforcementLearning/autoencoder/output/AutoencoderNet_(2,1,3,3)_[(3,16,1),(3,32,1)]_20_tictactoeAutoencoder_1000.pth')
    print ("autoencoderNN:\n{}".format(autoencoderNN))
    randomForest = BuildARandomForestDecoderFromAnAutoencoder(autoencoderNN, maximumNumberOfTrees=100, treesMaximumDepth=6)
    print ("randomForest=\n{}".format(randomForest))

    import tictactoe
    authority = tictactoe.Authority()
    positionTensorShape = authority.PositionTensorShape()
    randomPosition = torch.randn(5, positionTensorShape[0], positionTensorShape[1], positionTensorShape[2], positionTensorShape[3] )
    valuesList = randomForest.Value(randomPosition)
    print ("Decoder.py main(): valuesList = {}".format(valuesList))

    targetValues = torch.randn(5, 1)
    randomForest.LearnFromMinibatch(randomPosition, targetValues)
    valuesList = randomForest.Value(randomPosition)
    print("Decoder.py main(): After learning: valuesList = {}".format(valuesList))
    print("Decoder.py main(): targetValues = {}".format(targetValues))

    randomPosition = torch.randn(5, positionTensorShape[0], positionTensorShape[1], positionTensorShape[2],
                                 positionTensorShape[3])
    targetValues = torch.randn(5, 1)
    randomForest.LearnFromMinibatch(randomPosition, targetValues)
    valuesList = randomForest.Value(randomPosition)
    print("Decoder.py main(): After learning: valuesList = {}".format(valuesList))
    print("Decoder.py main(): targetValues = {}".format(targetValues))

if __name__ == '__main__':
    main()

