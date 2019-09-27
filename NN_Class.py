import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
sns.set()

class Prediction:
    """Predict the data based on given feature data"""

    def __init__(self, X_p, Y_p, Parameters):
        self.X_p = X_p
        self.Y_p = Y_p
        self.Parameters = Parameters

    def forward_prop(self):
        """
        The forward propagation function.
        Linear -> Relu for the all layer excluding last layer.
        Linear -> Sigmoid for the last layer.
        Args:
            X_p -- Data, numpy array
            Params -- Python list contains Weight and Bias matrixes

        Returns:
            Af_p -- Last activation value of the layer.
        """
        A = self.X_p
        L = len(self.Parameters) // 2

        for n in range(1, L):
            Z = np.dot(self.Parameters["W" + str(n)], A) + self.Parameters["b" + str(n)]
            A = np.maximum(0, Z)
            assert (Z.shape == (self.Parameters["W" + str(n)].shape[0], A.shape[1]))

        Z = np.dot(self.Parameters["W" + str(L)], A) + self.Parameters["b" + str(L)]
        Af_p = 1 / (1 + np.exp(-Z))
        assert (Z.shape == (self.Parameters["W" + str(L)].shape[0], Af_p.shape[1]))

        return Af_p

    def predict_f(self):
        """
        Predicts the results of a L-layer NN.

        Args:
            X_p -- Training data set
            Parameters -- Parameters of the trained model
            Y_p -- True target vector of data
        Returns:
            p -- Predictions for the given dataset X
            accuracy -- Accuracy of the the model on given data and parameters
        """

        m = self.X_p.shape[1]
        n = len(self.Parameters) // 2
        p = np.zeros((1, m))
        p = Prediction.forward_prop(self)
        p = np.round(p)
        acurracy = np.sum((p == self.Y_p)/m)
        return p, acurracy

class NNetwork(Prediction):
    """
    L-layer Neural Network
    """

    def __init__(self, X, Y, dims, learning_rate, Niter, print_cost=True):
        self.X = X
        self.Y = Y
        self.dims = dims
        self.learning_rate = learning_rate
        self.Niter = Niter
        self.print_cost= print_cost


    def initialize_parameters(self):
        """
        Initilize the parameters for neural network
        Args:
            X -- Data, numpy array
            dims -- Python list contains the dimensions of each layer in neural network
        Returns:
            params --   Python dictionary contains the initialized parameters of neural network
                        W[n] : Weight matrixes
                        b[n] : Bias Matrixes
        """
        np.random.seed(42)
        self.dims.insert(0, self.X.shape[0])
        params = {}


        for n in range(1, len(self.dims)):
            params["W" + str(n)] = np.random.randn(self.dims[n], self.dims[n-1]) * np.sqrt(2/self.dims[n-1])
            params["b" + str(n)] = np.zeros((self.dims[n], 1))

            assert(params["W" + str(n)].shape == (self.dims[n], self.dims[n-1]))
            assert(params["b" + str(n)].shape == (self.dims[n], 1))

        return params

    def forward_propagation(self, params):
        """
        The forward propagation function.
        Linear -> Relu for the all layer excluding last layer.
        Linear -> Sigmoid for the last layer.
        Args:
            X -- Data, numpy array
            Params -- Python list contains initialized Weight and Bias matrixes

        Returns:
            Predict -- Last activation value of the layer.
            Caches -- Neural Network parameters for backward propagation (A,Z,W,b)
        """
        caches = []
        A = self.X
        L = len(params) // 2

        for n in range(1, L):
            Z = np.dot(params["W" + str(n)], A) + params["b" +str(n)]
            caches.append((A, Z, params["W" + str(n)], params["b" +str(n)]))
            A = np.maximum(0,Z)
            assert(Z.shape == (params["W" + str(n)].shape[0], A.shape[1]))

        Z = np.dot(params["W" + str(L)], A) + params["b" +str(L)]
        Af = 1/ (1 + np.exp(-Z))   #Last Layer
        caches.append((A, Z, params["W" + str(L)], params["b" +str(L)]))
        assert (Z.shape == (params["W" + str(L)].shape[0], Af.shape[1]))

        return Af, caches

    def Cost_func(self, Af):
        """
        The cross-entropy cost function.

        Args:
            Af -- Probability vector of final forward layer predictions
            Y -- True target vector of data
        Returns:
            const -- cross-entropy cost
        """
        m = self.Y.shape[1]
        cost = -1/m * np.sum(np.multiply(self.Y, np.log(Af)) + np.multiply((1-self.Y), np.log(1-Af)))
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    def backward_propagation(self, Af, caches):
        """
        The backward propagation function.
        Args:
            Af -- Probability vector of final forward layer predictions
            Y -- True target vector of data
            caches -- tuple values of (A,Z,W,b) from forward propagation

        Returns:
            Grads -- Gradients Dictionary of dA, dW, db
        """
        m = Af.shape[1]
        L = len(caches) #number of layer
        grads = {}
        Y = self.Y.reshape(Af.shape)
        dAf = -(np.divide(self.Y, Af) - np.divide(1-Y, 1-Af)) # initialization of backward propagation

        for n in reversed(range(L)):
            if n == L-1:
                # sigmoid backward
                sigmoid_cache = caches[-1]
                A, Z, W, b = sigmoid_cache
                m = A.shape[1]
                dZ = dAf * (1 / (1 + np.exp(-Z))) * (1 - 1 / (1 + np.exp(-Z)))
                grads["dW" + str(n+1)] = 1 / m * np.dot(dZ, A.T)
                grads["db" + str(n+1)] = np.sum(dZ, axis=1).reshape(-1, 1) / m
                grads["dA" + str(n)] = np.dot(W.T, dZ)
            else:
                relu_cache = caches[n]
                A, Z, W, b = relu_cache
                m = A.shape[1]

                dZ = np.array(grads["dA" + str(n+1)], copy=True)
                dZ[Z <= 0]=0
                grads["dW" + str(n+1)] = 1 / m * np.dot(dZ, A.T)
                grads["db" + str(n+1)] = 1./m * np.sum(dZ, axis = 1, keepdims = True)
                grads["dA" + str(n)] = np.dot(W.T, dZ)
            assert (dZ.shape == Z.shape)
            assert (grads["dW" + str(n+1)].shape == W.shape)
            assert (grads["db" + str(n+1)].shape == b.shape)
            assert (grads["dA" + str(n)].shape == A.shape)

        return grads

    def parameters_update(self, params, grads):
        """
        Update the parameters using gradient descent
        Args:
            params -- Python dictionary containing parameters, Weight and Bias matrixes
            grads -- Python dictionary containing gradients, output of backward propagation
                    dW,db,dA matrixes
            learning_rate -- Learning rate (float)
        Returns:
            params -- Python dictionary containing updated parameters, Weight and Bias matrixes
        """
        L = len(params) // 2 # Number of layers
        for n in range(L):
            params["W" + str(n +1)] = params["W" + str(n +1)] - self.learning_rate * grads["dW" + str(n +1)]
            params["b" + str(n + 1)] = params["b" + str(n + 1)] - self.learning_rate * grads["db" + str(n + 1)]

        return params

    def NN_image(self):
        """
        Implements L-layer neural network : (Linear -> Relu * L-1) * (L-1) -> Linear -> Sigmoid
        Args:
            X -- Training data set
            Y -- True target vector of data
            layer_dimensions --  List of dimensions of the layer
            learning_rate -- learning rate of gradient descent
            Niter -- Numbe rof iterations of the optimization loop
            print_cost -- True default, prints the cost every 150 iterations
        Returns:
            Parameters -- Python dictionary of optimized Weight and bias matrixes
        """
        costs = []
        np.random.seed(42)
        params = NNetwork.initialize_parameters(self)

        for i in range(0, self.Niter):

            # [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
            Af, caches = NNetwork.forward_propagation(self, params)
            cost = NNetwork.Cost_func(self, Af)
            grads = NNetwork.backward_propagation(self, Af, caches)
            parameters = NNetwork.parameters_update(self, params, grads)

            if self.print_cost & (i % 100 == 0) :
                print("Cost after iteration {}: {}".format(i, cost))
                costs.append(cost)

        plt.plot(np.squeeze(costs))
        plt.title("Learning rate = " +  str(self.learning_rate))
        plt.ylabel("Cost")
        plt.xlabel("Iterations (x100)")
        plt.show()

        return parameters

    @staticmethod
    def predict(X, Y, Parameters):
        """
        Calls  predict_ f method on Prediction class for given data

        Args:
            X -- Data, numpy array
            Y -- True target vector of data
            Parameter -- Python list contains Weight and Bias matrixes
        Returns:
            prediction -- Predictions for the given dataset X
            accuracy -- Accuracy of the the model on given data and parameters

        """
        predict = Prediction(X, Y, Parameters)
        predictions, accuracy = predict.predict_f()
        return predictions, accuracy



def get_Data(train, test):
    train_x = np.array(train["train_set_x"])
    train_x = train_x.reshape(-1, train_x.shape[0])
    train_y = np.array(train["train_set_y"])
    train_y = train_y.reshape(1,train_y.shape[0])
    test_x = np.array(test["test_set_x"])
    test_x = test_x.reshape(-1, test_x.shape[0])
    test_y = np.array(test["test_set_y"])
    test_y = test_y.reshape(1, test_y.shape[0])
    classes = np.array(train["list_classes"])
    train_x_flat = train_x / 255.
    test_x_flat = test_x / 255.
    return train_x_flat, test_x_flat, train_y, test_y


train_dataset = h5py.File("../train_catvnoncat.hdf5", "r")
test_dataset = h5py.File("../test_catvnoncat.hdf5", "r")

X_train, X_test, Y_train, Y_test= get_Data(train_dataset, test_dataset)
layer_dimensions = [7, 5, 4, 4, 1]
P = NNetwork(X_train, Y_train, layer_dimensions, learning_rate = 0.006, Niter=2000)
parameters = P.NN_image()

predictions, train_accuracy = P.predict(X_train, Y_train, parameters)
print("Train Accuracy: ", train_accuracy)
predictions, test_accuracy = P.predict(X_test, Y_test, parameters)
print("Test Accuracy: ", test_accuracy)










