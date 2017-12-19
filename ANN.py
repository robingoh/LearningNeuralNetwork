# Final project for CS 156
# Author: Robin Goh (Wang Foo)
# This assignment is done by referencing to the book Make Your Own Neural Network by Tariq Rashid.
import numpy
# scipy.special for the sigmoid function expit(), not used since I am implementing the sigmoid function.
import scipy.special

# A neural network class that can create, train and query a 3-layer neural networks.
# I was able to achieve average correct percentage of 0.94 (94 %) with this neural networks.


def mul(a, b):
    a_row_len = len(a)
    a_col_len = len(a[0])
    b_row_len = len(b)
    b_col_len = len(b[0])

    result = numpy.zeros((a_row_len, b_col_len))
    # iterate through rows of a
    for i in range(a_row_len):
        # iterate through cols of b
        for j in range(b_col_len):
            # iterate through rows of b
            for k in range(b_row_len):
                result[i][j] += a[i][k] * b[k][j]
    return result


# Transpose is tested by using numpy.dot() in the neural network and the neural network works correctly.
# I found out that using numpy.transpose() is approximately 8 seconds faster than my definition of transpose(). 
def transpose(a):
    num_of_rows = len(a)
    num_of_cols = len(a[0])
    transposed_matrix = numpy.zeros((num_of_cols, num_of_rows))
    for j in range(num_of_cols):
        for i in range(num_of_rows):
            transposed_matrix[j][i] = a[i][j]
    return transposed_matrix


# dot product is not working and I was not able to find out what went wrong. 
# In the end, I use the numpy.dot() for the neural network.  
def dot(a, b):
    return mul(transpose(a), b)


# neural network class definition
class NeuralNetwork:
    # initialize the neural network
    # need number of input, hidden and out layer nodes.
    # also the learning rate
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inputNodes = input_nodes
        self.hiddenNodes = hidden_nodes
        self.outputNodes = output_nodes
        self.learningRate = learning_rate

        # Most important part of the network is the link weights
        # since they're used to calculate the forwarded signal and the backwarded error.
        # Weights can be expressed in a matrix. So create:
        # 1. a matrix for the weights of links between the input and hidden layers
        #    with size hiddenNodes * inputNodes
        # 2. a matrix for the weights of links between the hidden and output layers
        #    with size outputNodes * hiddenNodes
        # Initially, the values of the link weights should be small and random.
        # Use numpy.random.rand(rows, cols) to generate an array between 0 and 1 randomly.
        # The weights can be negative, so subtract the above with 0.5 to get range -0.5 to 0.5
        # weight matrices wih, who
        # w_ij is the weight of the link from node i to node j in the next layer
        self.wih = numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.who = numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5
        # To optimize, use numpy.random.normal() to sample a normal distribution
        # params are the center of the distribution, standard deviation, and size of a numpy array
        # self.wih = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        # self.who = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        # activation function is the sigmoid function
        self.activationFunction = lambda x: 1 / (1 + numpy.exp(-x))
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # 1st part: working out the output of an input, just like in query()
        # convert arguments from list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)

        # 2nd part: taking the calculated output from the 1st part,
        #           comparing it with the desired output,
        #           and using the difference to update the network weights.
        # error = target - actual
        output_errors = targets - final_outputs
        # error_hidden = transpose(weights_hidden_output) numpy.dot errors_output
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # to refine the weights at each layer,
        # use output_errors for the weights between the hidden and final layers
        # use hidden_errors for the weights between the input and hidden layers
        # Expression for updating the weight of the link between a node j and node k in the next layer:
        # delW_jk = alpha * E_k * sigmoid(O_k) * ( 1- sigmoid(O_k) ) dot transpose(O_j), alpha is the learning rate
        # update the weights of the links between the hidden and output layers
        self.who += self.learningRate \
                    * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), transpose(hidden_outputs))
        # update the weights of the links between the input and hidden layers
        self.wih += self.learningRate \
                    * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), transpose(inputs))

        pass

    # query the nerural network
    def query(self, inputs_list):
        # convert inputs list to 2d array, since input is written inside square brackets, which will be a list
        inputs = numpy.array(inputs_list, ndmin=2).T
        # signals into the hidden layer nodes = weights of the link between input_hidden layers numpy.dot input matrix
        # calculate the signals into the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # to get the signals emerging from the hidden node, apply the sigmoid squashing function to each emergin signals
        # O_hidden = sigmoid(X_hidden)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)
        # calculate the signals into the final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from the final output layer
        final_outputs = self.activationFunction(final_inputs)
        return final_outputs


if __name__ == '__main__':
    # define number of input, hidden and output nodes, and the learning rate
    inputNodes = 784
    hiddenNodes = 100
    outputNodes = 10
    learningRate = 0.3

    # create an instance of the neural network
    ann = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

    # load the mnist training data CSV file into a list
    trainingDataFile = open("/Users/robg/Downloads/mnist_train.csv", 'r')
    trainingDataList = trainingDataFile.readlines()
    trainingDataFile.close()

    # train the neural network
    for entry in trainingDataList:
        # split the data separated by ','
        data = entry.split(',')

        # scale and shift the inputs
        # Rescale input color values from [0, 255] to [0.01, 1.0]. 0.01 is chosen since zero valued inputs can
        # kill weight updates. No need to choose 0.99 as the upper end of the input but should avoid in the output case.
        trainingInputs = (numpy.asfarray(data[1:]) / 255.0 * 0.99) + 0.01

        # create the target output values with all 0.01 except the desired label should be 0.99
        targetOutputs = numpy.zeros(outputNodes) + 0.01
        targetOutputs[int(data[0])] = 0.99

        ann.train(trainingInputs, targetOutputs)
        pass

    # test the trained neural network
    testDataFile = open("/Users/robg/Downloads/mnist_test.csv", 'r')
    testDataList = testDataFile.readlines()
    testDataFile.close()

    scores = []
    for entry in testDataList:
        testData = entry.split(',')
        # correct label is the first element in the data
        correctLabel = int(testData[0])
        print("a. correct label      : ", correctLabel)
        # scale and shift the test inputs
        testInputs = (numpy.asfarray(testData[1:]) / 255.0 * 0.99) + 0.01
        # query the trained neural network
        outputs = ann.query(testInputs)
        # get the highest value of index which corresponds to the most likely label
        actualLabel = numpy.argmax(outputs)
        print("b. neural net's answer: ", actualLabel)
        if actualLabel == correctLabel:
            scores.append(1)
        else:
            scores.append(0)
        pass

    # calculate the correct percentage
    scores_array = numpy.asarray(scores)
    print("correct percentage = ", scores_array.sum() / scores_array.size)
