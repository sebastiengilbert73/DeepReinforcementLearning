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

    def LatentVariables(self, positionBatch):
        minibatchSize = positionBatch.shape[0]
        activationTensor = positionBatch
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        latentVariablesTensor = self.ReLU(activationTensor)
        # print ("RandomForest.LatentVariables(): latentVariablesTensor = \n{}".format(latentVariablesTensor))
        # print("RandomForest.LatentVariables(): latentVariablesTensor.shape = {}".format(latentVariablesTensor.shape))

        # Create products of latent variables
        #productsTensor = self.ProductsTensor(latentVariablesTensor)
        # print ("LatentVariables(): productsTensor.shape =\n{}".format(productsTensor.shape))
        #latentVariablesTensor = torch.cat((latentVariablesTensor, productsTensor), 1)  # Concatenate the products of latent variables
        return latentVariablesTensor

    def Value(self, positionBatch):
        # Compute latent variables
        minibatchSize = positionBatch.shape[0]

        """activationTensor = positionBatch
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        latentVariablesTensor = self.ReLU(activationTensor)
        #print ("RandomForest.Value(): latentVariablesTensor = \n{}".format(latentVariablesTensor))
        #print("RandomForest.Value(): latentVariablesTensor.shape = {}".format(latentVariablesTensor.shape))

        # Create products of latent variables
        productsTensor = self.ProductsTensor(latentVariablesTensor)
        #print ("Value(): productsTensor.shape =\n{}".format(productsTensor.shape))
        latentVariablesTensor = torch.cat((latentVariablesTensor, productsTensor), 1) # Concatenate the products of latent variables
        """
        latentVariablesTensor = self.LatentVariables(positionBatch)

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

        """activationTensor = minibatchFeaturesTensor
        activationTensor = self.encodingBodyStructureSeq(activationTensor)
        activationTensor = activationTensor.view(minibatchSize, self.encodingBodyActivationNumberOfEntries)
        activationTensor = self.encodingFullyConnectedLayer(activationTensor)
        latentVariablesTensor = self.ReLU(activationTensor)
        """
        latentVariablesTensor = self.LatentVariables(minibatchFeaturesTensor)

        while len(self.treesList) >= self.maximumNumberOfTrees: # Remove a new tree
            treeErrorSumsList = []
            for treeNdx in range(len(self.treesList)):
                treeErrorSum = 0
                for exampleNdx in range(minibatchSize):
                    featuresTensor = latentVariablesTensor[exampleNdx]
                    prediction = self.treesList[treeNdx].predict(featuresTensor.detach().numpy().reshape(1, -1))[0]
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

    def ProductsTensor(self, latentVariablesTensor):
        #print ("ProductsTensor(): latentVariablesTensor.shape = {}".format(latentVariablesTensor.shape))
        latentVariablesTensorShape = latentVariablesTensor.shape
        minibatchSize = latentVariablesTensorShape[0]
        numberOfLatentVariables = latentVariablesTensorShape[1]
        numberOfProducts = numberOfLatentVariables * (numberOfLatentVariables - 1)//2 # (2 in N)
        productsTensor = torch.zeros(minibatchSize, numberOfProducts)

        for exampleNdx in range(minibatchSize):
            runningRowNdx = 0
            for number1Ndx in range(numberOfLatentVariables):
                number1 = latentVariablesTensor[exampleNdx, number1Ndx].item()
                for number2Ndx in range(number1Ndx + 1, numberOfLatentVariables):
                    number2 = latentVariablesTensor[exampleNdx, number2Ndx].item()
                    product = number1 * number2
                    productsTensor[exampleNdx, runningRowNdx] = product
                    runningRowNdx += 1
        return productsTensor


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

    import tictactoe
    import pickle
    import utilities

    randomForest = Predictor.Load('/home/sebastien/projects/DeepReinforcementLearning/positionEvaluation/outputs/tictactoe_23.bin')

    print (randomForest.encodingBodyStructureSeq)
    authority = tictactoe.Authority()
    playersList = authority.PlayersList()

    inputTensor = authority.InitialPosition()
    #inputTensor[1, 0, 0, 0] = 1
    #inputTensor[0, 0, 0, 1] = 1
    #inputTensor[0, 0, 0, 2] = 1
    #inputTensor[0, 0, 1, 0] = 1
    inputTensor[1, 0, 1, 1] = 1
    #inputTensor[1, 0, 1, 2] = 1
    #inputTensor[0, 0, 2, 0] = 1
    #inputTensor[1, 0, 2, 1] = 1
    #inputTensor[0, 0, 2, 2] = 1
    authority.Display(inputTensor)

    value = randomForest.Value(inputTensor.unsqueeze(0))
    print ("value = {}".format(value))

    legalMoveTensorsList = utilities.LegalMoveTensorsList(authority, inputTensor, playersList[0])
    print ("legalMoveTensorsList = {}".format(legalMoveTensorsList))

    """legalMoveToExpectedRewardDict = Predictor.LegalMoveToExpectedReward(
        randomForest, authority, inputTensor, playersList[0], numberOfGames=10, epsilon=0.1)

    print ("legalMoveToExpectedRewardDict = {}".format(legalMoveToExpectedRewardDict))
    """

    """positionsList, winner = Predictor.SimulateAGame(randomForest, authority, inputTensor, playersList[1])
    print ("positionsList = \n{}".format(positionsList))
    print ("winner = {}".format(winner))
    """

    (numberOfWinsForEvaluator, numberOfWinsForRandomPlayer, numberOfDraws) = Predictor.SimulateGamesAgainstARandomPlayer(
        randomForest, authority, 30
    )
    print ("(numberOfWinsForEvaluator, numberOfWinsForRandomPlayer, numberOfDraws) = ({}, {}, {})".format(numberOfWinsForEvaluator, numberOfWinsForRandomPlayer, numberOfDraws))


if __name__ == '__main__':
    main()

