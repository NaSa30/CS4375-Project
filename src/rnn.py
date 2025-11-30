#numpy used to create the RNN from scratch
import numpy as np

# convert_to_numpy() :this function onverts the input to a numpy array
def convert_to_numpy(x):
    return np.array(x)

class RNN:
    #objective is to calc and initalize the required variables
    def __init__(self, inputSize, hiddenSize, outputSize, length, learnRate, seed=None):
        # check if seed is empty
        if seed is not None:
            np.random.seed(seed)

        # Assign the inputted variables to the their corresponding class variables
        self.inputSize = inputSize # x
        self.hiddenSize = hiddenSize  # h
        self.outputSize = outputSize  #y
        self.length = length
        self.learnRate = learnRate
        self.seed = seed

        # Calculate weight matrices
        self.weight_xh = np.random.randn(hiddenSize, inputSize) * 0.01 # Weight of connection between input layer and hidden lauer
        self.weight_hh = np.random.randn(hiddenSize, hiddenSize) * 0.01 # Weight of connection between previous and current hidden layer
        self.weight_hy = np.random.randn(outputSize, hiddenSize) * 0.01 # Weight of connection between hidden layer and output layer

        # Biases Vectors
        self.bias_h = np.zeros((hiddenSize, 1)) # Bias for hidden layer
        self.bias_y = np.zeros((outputSize, 1)) # Bias for output lauer

    # forward() : Forward pass throughout time
    def forward(self, input):
        input = convert_to_numpy(input)
        self.hiddenStates = {}
        hiddenStates_prev = np.zeros((self.hiddenSize, 1))
        self.hiddenStates[-1] = hiddenStates_prev

        # Append into hiddenStates while passing through the time
        for i in range(self.length):
            currInput_time = input[i].reshape(-1, 1) # Current input data at time i
            hiddenStates_prev = np.tanh(self.weight_xh @ currInput_time + self.weight_hh @ hiddenStates_prev + self.bias_h)
            self.hiddenStates[i] = hiddenStates_prev

        return self.weight_hy @ hiddenStates_prev + self.bias_y
    
    ### backward() : Backpropagation through time to update weigts in place by using SGD per sample
    def backward(self, input, target):
        input = convert_to_numpy(input)
        target = convert_to_numpy(target).reshape(self.outputSize, 1)

        # Use foreward pass
        predOutput = self.forward(input)
        loss = 0.5 * (predOutput - target) ** 2

        # Gradients
        #- Weights:
        gradient_wxh = np.zeros_like(self.weight_xh)
        gradient_whh = np.zeros_like(self.weight_hh)
        gradient_why = np.zeros_like(self.weight_hy)
        #- Biases:
        gradient_bh = np.zeros_like(self.bias_h)
        gradient_by = np.zeros_like(self.bias_y)

        #- Output
        gradient_why += (predOutput - target) @ self.hiddenStates[self.length - 1].T
        gradient_by += predOutput - target
        gradient_hnext = self.weight_hy.T @ (predOutput - target)

        # Backpropagation through time
        for i in reversed(range(self.length)):
            hiddenLayer = self.hiddenStates[i]
            prev_hiddenLayer = self.hiddenStates[i - 1]

            gradient_tanh = (1 - hiddenLayer ** 2) * gradient_hnext
            gradient_bh += gradient_tanh
            gradient_wxh += gradient_tanh @ input[i].reshape(1, -1)
            gradient_whh += gradient_tanh @ prev_hiddenLayer.T
            gradient_hnext = self.weight_hh.T @ gradient_tanh
        
        # Gradient clipping
        for i in [gradient_wxh, gradient_whh, gradient_why, gradient_bh, gradient_by]:
            np.clip(i, -1.0, 1.0, out=i)

        # Update weight
        self.weight_xh -= self.learnRate * gradient_wxh
        self.weight_hh -= self.learnRate * gradient_whh
        self.weight_hy -= self.learnRate * gradient_why
        self.bias_h -= self.learnRate * gradient_bh
        self.bias_y -= self.learnRate * gradient_by

        return float(loss)
    
    # train_from_data() : Uses sample by sample SGD to train data from data_loader.py
    def train_from_data(self, data, epochs, verbose):
        losses = []

        # Calculate epoch average losses
        for i in range(epochs):
            total = 0.0
            count = 0

            for x, y in data:
                xnp = convert_to_numpy(x)
                ynp = convert_to_numpy(y)

                for j in range(xnp.shape[0]):
                    loss = self.backward(xnp[j], ynp[j])
                    total += loss
                    count += 1

            # it said to fix this indentation
            average = total / max(1, count)
            losses.append(average)

            if (i + 1) % verbose == 0:
                print(f"Epoch {i + 1}/{i} average loss: {average:.3f}")

        return losses
    
    ### predict() : Returns prediction
    def predict(self, input):
        prediction = self.forward(input)
        return float(np.squeeze(prediction))
    
    ### predict_from_data() : Predict results from using data_loarder.py
    def predict_from_data(self, data):
        prediction = []
        targets = []

        for x, y in data:
            xnp = convert_to_numpy(x)
            ynp = convert_to_numpy(y)

            for i in range(xnp.shape[0]):
                prediction.append(self.predict(xnp[i]))
            
            targets.extend(ynp.reshape(-1).tolist())
        
        return np.array(prediction), np.array(targets)
    
    ### save() : Save serveral arrays into copressed files by numpy
    
    ### load() : Initialize internal state by numpy
    def load(self, path):
        data = np.load(path)
        self.weight_xh = data['weight_xh']
        self.weight_hh = data['weight_hh']
        self.weight_hy = data['weight_hy']
        self.bias_h = data['bias_h']
        self.bias_y = data['bias_y']

    def save(self, path):
        np.savez_compressed(path, weight_xh = self.weight_xh, weight_hh = self.weight_hh, weight_hy = self.weight_hy, bias_h = self.bias_h, bias_y = self.bias_y)

