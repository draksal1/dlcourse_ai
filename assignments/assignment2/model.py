import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg

        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size,n_output)]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for key, param in self.params().items():
            param.grad = np.zeros_like(param.grad)
        pred = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            pred = layer.forward(pred)
        loss, pred = softmax_with_cross_entropy(pred, y)
        grad = self.layers[-1].backward(pred)
        for layer in self.layers[::-1][1:]:
            grad = layer.backward(grad)
        for key, param in self.params().items():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            pred = layer.forward(pred)

        pred = softmax(pred)
        pred = np.argmax(pred, axis=1)

        return pred

    def params(self):
        result = {}
        for ind, layer in enumerate(self.layers):
            for param in layer.params().items():
                result[str(ind) + '_' + param[0]] = param[1]

        return result
