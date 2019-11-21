import abc

class Evaluator(abc.ABC):
    """
    Abstract class that predicts the value of a position, when it is the opponent's turn
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def Value(self, position):
        pass # return a float

    @abc.abstractmethod
    def LearnFromMinibatch(self, minibatchFeaturesTensor, minibatchTargetValues):
        pass