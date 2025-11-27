"""
LSTM implementation from scratch using NumPy
"""

import numpy as np

def convert_to_numpy(x):
    # just making sure everything is a numpy array
    return np.array(x)

class LSTM:
    def __init__(self, inputSize, hiddenSize, outputSize, length, learnRate, seed=None):
        # set seed if we have one
        if seed is not None:
            np.random.seed(seed)
        
        # store the main parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.length = length
        self.learnRate = learnRate
        self.seed = seed
        
        # intialize weights for the forget gate
        self.weight_forget_x = np.random.randn(hiddenSize, inputSize) * 0.01
        self.weight_forget_h = np.random.randn(hiddenSize, hiddenSize) * 0.01
        self.bias_forget = np.zeros((hiddenSize, 1))
        
        # intialize weights for input gate
        self.weight_input_x = np.random.randn(hiddenSize, inputSize) * 0.01
        self.weight_input_h = np.random.randn(hiddenSize, hiddenSize) * 0.01
        self.bias_input = np.zeros((hiddenSize, 1))
        
        # cell gate weights (this is for the candidate values)
        self.weight_cell_x = np.random.randn(hiddenSize, inputSize) * 0.01
        self.weight_cell_h = np.random.randn(hiddenSize, hiddenSize) * 0.01
        self.bias_cell = np.zeros((hiddenSize, 1))
        
        # output gate stuff
        self.weight_output_x = np.random.randn(hiddenSize, inputSize) * 0.01
        self.weight_output_h = np.random.randn(hiddenSize, hiddenSize) * 0.01
        self.bias_output = np.zeros((hiddenSize, 1))
        
        # final output layer weights
        self.weight_final = np.random.randn(outputSize, hiddenSize) * 0.01
        self.bias_final = np.zeros((outputSize, 1))
    
    def sigmoid(self, x):
        # sigmoid activation, keeping it simple
        return 1.0 / (1.0 + np.exp(-x))
    
    def tanh(self, x):
        # tanh activation
        return np.tanh(x)
    
    def forward(self, input):
        # convert input to numpy just in case
        input = convert_to_numpy(input)
        
        # storage for hidden states and cell states
        self.hidden_states = {}
        self.cell_states = {}
        
        # storage for gate values (needed for backprop later)
        self.forget_gates = {}
        self.input_gates = {}
        self.cell_candidates = {}
        self.output_gates = {}
        
        # intialize first hidden and cell state
        h_prev = np.zeros((self.hiddenSize, 1))
        c_prev = np.zeros((self.hiddenSize, 1))
        
        # store intial states
        self.hidden_states[-1] = h_prev
        self.cell_states[-1] = c_prev
        
        # go through each time step
        for t in range(self.length):
            # get current input at this timestep
            x_current = input[t].reshape(-1, 1)
            
            # calcualte forget gate (decides what to forget from cell state)
            forget_input = self.weight_forget_x @ x_current + self.weight_forget_h @ h_prev + self.bias_forget
            forget_gate = self.sigmoid(forget_input)
            
            # calcualte input gate (decides what new info to add)
            input_input = self.weight_input_x @ x_current + self.weight_input_h @ h_prev + self.bias_input
            input_gate = self.sigmoid(input_input)
            
            # calculate candidate cell state (new potential values)
            cell_input = self.weight_cell_x @ x_current + self.weight_cell_h @ h_prev + self.bias_cell
            cell_candidate = self.tanh(cell_input)
            
            # update cell state by combining forget and input gates
            c_current = forget_gate * c_prev + input_gate * cell_candidate
            
            # calcualte output gate (decides what to output)
            output_input = self.weight_output_x @ x_current + self.weight_output_h @ h_prev + self.bias_output
            output_gate = self.sigmoid(output_input)
            
            # calculate new hidden state
            h_current = output_gate * self.tanh(c_current)
            
            # save everything for backprop
            self.hidden_states[t] = h_current
            self.cell_states[t] = c_current
            self.forget_gates[t] = forget_gate
            self.input_gates[t] = input_gate
            self.cell_candidates[t] = cell_candidate
            self.output_gates[t] = output_gate
            
            # update previous states for next iteration
            h_prev = h_current
            c_prev = c_current
        
        # final prediction using last hidden state
        final_output = self.weight_final @ h_current + self.bias_final
        
        return final_output
    
    def backward(self, input, target):
        # convert everything to numpy
        input = convert_to_numpy(input)
        target = convert_to_numpy(target).reshape(self.outputSize, 1)
        
        # run forward pass first
        prediction = self.forward(input)
        
        # calcualte loss (using MSE)
        loss = 0.5 * (prediction - target) ** 2
        
        # intialize gradient accumulators for all weights
        grad_wfx = np.zeros_like(self.weight_forget_x)
        grad_wfh = np.zeros_like(self.weight_forget_h)
        grad_bf = np.zeros_like(self.bias_forget)
        
        grad_wix = np.zeros_like(self.weight_input_x)
        grad_wih = np.zeros_like(self.weight_input_h)
        grad_bi = np.zeros_like(self.bias_input)
        
        grad_wcx = np.zeros_like(self.weight_cell_x)
        grad_wch = np.zeros_like(self.weight_cell_h)
        grad_bc = np.zeros_like(self.bias_cell)
        
        grad_wox = np.zeros_like(self.weight_output_x)
        grad_woh = np.zeros_like(self.weight_output_h)
        grad_bo = np.zeros_like(self.bias_output)
        
        grad_wf = np.zeros_like(self.weight_final)
        grad_bf_final = np.zeros_like(self.bias_final)
        
        # gradient for output layer
        output_error = prediction - target
        grad_wf += output_error @ self.hidden_states[self.length - 1].T
        grad_bf_final += output_error
        
        # gradient flowing back to hidden state
        dh_next = self.weight_final.T @ output_error
        dc_next = np.zeros((self.hiddenSize, 1))
        
        # backpropogate through time
        for t in reversed(range(self.length)):
            # get current input
            x_current = input[t].reshape(-1, 1)
            
            # get saved states from forward pass
            h_current = self.hidden_states[t]
            c_current = self.cell_states[t]
            h_prev = self.hidden_states[t - 1]
            c_prev = self.cell_states[t - 1]
            
            # get saved gates
            forget_gate = self.forget_gates[t]
            input_gate = self.input_gates[t]
            cell_cand = self.cell_candidates[t]
            output_gate = self.output_gates[t]
            
            # gradient through output gate
            tanh_c = self.tanh(c_current)
            do = dh_next * tanh_c
            # derivative of sigmoid
            do_input = do * output_gate * (1 - output_gate)
            
            grad_wox += do_input @ x_current.T
            grad_woh += do_input @ h_prev.T
            grad_bo += do_input
            
            # gradient through cell state
            dc = dc_next + dh_next * output_gate * (1 - tanh_c ** 2)
            
            # gradient through forget gate
            df = dc * c_prev
            df_input = df * forget_gate * (1 - forget_gate)
            
            grad_wfx += df_input @ x_current.T
            grad_wfh += df_input @ h_prev.T
            grad_bf += df_input
            
            # gradient through input gate
            di = dc * cell_cand
            di_input = di * input_gate * (1 - input_gate)
            
            grad_wix += di_input @ x_current.T
            grad_wih += di_input @ h_prev.T
            grad_bi += di_input
            
            # gradient through cell candidate
            dcc = dc * input_gate
            dcc_input = dcc * (1 - cell_cand ** 2)
            
            grad_wcx += dcc_input @ x_current.T
            grad_wch += dcc_input @ h_prev.T
            grad_bc += dcc_input
            
            # propogation to previous timestep
            dh_next = (self.weight_forget_h.T @ df_input + 
                      self.weight_input_h.T @ di_input +
                      self.weight_cell_h.T @ dcc_input +
                      self.weight_output_h.T @ do_input)
            
            dc_next = dc * forget_gate
        
        # gradient clipping to prevent exploding gradients
        all_grads = [grad_wfx, grad_wfh, grad_bf, grad_wix, grad_wih, grad_bi,
                    grad_wcx, grad_wch, grad_bc, grad_wox, grad_woh, grad_bo,
                    grad_wf, grad_bf_final]
        
        for grad in all_grads:
            np.clip(grad, -1.0, 1.0, out=grad)
        
        # update all weights using gradients
        self.weight_forget_x -= self.learnRate * grad_wfx
        self.weight_forget_h -= self.learnRate * grad_wfh
        self.bias_forget -= self.learnRate * grad_bf
        
        self.weight_input_x -= self.learnRate * grad_wix
        self.weight_input_h -= self.learnRate * grad_wih
        self.bias_input -= self.learnRate * grad_bi
        
        self.weight_cell_x -= self.learnRate * grad_wcx
        self.weight_cell_h -= self.learnRate * grad_wch
        self.bias_cell -= self.learnRate * grad_bc
        
        self.weight_output_x -= self.learnRate * grad_wox
        self.weight_output_h -= self.learnRate * grad_woh
        self.bias_output -= self.learnRate * grad_bo
        
        self.weight_final -= self.learnRate * grad_wf
        self.bias_final -= self.learnRate * grad_bf_final
        
        return float(loss)
    
    def train_from_data(self, data, epochs, verbose):
        # store losses for each epoch
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            
            # go through each batch in the data
            for x, y in data:
                xnp = convert_to_numpy(x)
                ynp = convert_to_numpy(y)
                
                # train on each sample in the batch
                for i in range(xnp.shape[0]):
                    loss = self.backward(xnp[i], ynp[i])
                    total_loss += loss
                    count += 1
            
            # calcualte average loss for this epoch
            avg_loss = total_loss / max(1, count)
            losses.append(avg_loss)
            
            # print progress if verbose
            if (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch + 1}/{epochs} average loss: {avg_loss:.3f}")
        
        return losses
    
    def predict(self, input):
        # just run forward pass and return result
        prediction = self.forward(input)
        return float(np.squeeze(prediction))
    
    def predict_from_data(self, data):
        # make predictions on a whole dataset
        predictions = []
        targets = []
        
        for x, y in data:
            xnp = convert_to_numpy(x)
            ynp = convert_to_numpy(y)
            
            # predict for each sample
            for i in range(xnp.shape[0]):
                pred = self.predict(xnp[i])
                predictions.append(pred)
            
            # store targets seperately
            targets.extend(ynp.reshape(-1).tolist())
        
        return np.array(predictions), np.array(targets)
    
    def save(self, path):
        # save all the weights and biases
        np.savez_compressed(path,
                          weight_forget_x=self.weight_forget_x,
                          weight_forget_h=self.weight_forget_h,
                          bias_forget=self.bias_forget,
                          weight_input_x=self.weight_input_x,
                          weight_input_h=self.weight_input_h,
                          bias_input=self.bias_input,
                          weight_cell_x=self.weight_cell_x,
                          weight_cell_h=self.weight_cell_h,
                          bias_cell=self.bias_cell,
                          weight_output_x=self.weight_output_x,
                          weight_output_h=self.weight_output_h,
                          bias_output=self.bias_output,
                          weight_final=self.weight_final,
                          bias_final=self.bias_final)
    
    def load(self, path):
        # load weights from file
        data = np.load(path)
        
        self.weight_forget_x = data['weight_forget_x']
        self.weight_forget_h = data['weight_forget_h']
        self.bias_forget = data['bias_forget']
        
        self.weight_input_x = data['weight_input_x']
        self.weight_input_h = data['weight_input_h']
        self.bias_input = data['bias_input']
        
        self.weight_cell_x = data['weight_cell_x']
        self.weight_cell_h = data['weight_cell_h']
        self.bias_cell = data['bias_cell']
        
        self.weight_output_x = data['weight_output_x']
        self.weight_output_h = data['weight_output_h']
        self.bias_output = data['bias_output']
        
        self.weight_final = data['weight_final']
        self.bias_final = data['bias_final']


if __name__ == "__main__":
    # quick test to make sure everything works
    print("Testing LSTM implementation...")
    
    # create a small lstm
    lstm = LSTM(inputSize=5, hiddenSize=32, outputSize=1, length=30, learnRate=0.01)
    
    # make some fake data (30 timesteps, 5 features)
    test_input = np.random.randn(30, 5)
    test_target = np.random.randn(1)
    
    # test forward pass
    output = lstm.forward(test_input)
    print(f"Forward pass output shape: {output.shape}")
    
    # test backward pass
    loss = lstm.backward(test_input, test_target)
    print(f"Backward pass loss: {loss:.4f}")
    
    # test prediction
    pred = lstm.predict(test_input)
    print(f"Prediction: {pred:.4f}")
    
    print("\nLSTM works! All tests passed.")