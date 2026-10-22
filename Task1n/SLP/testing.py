import numpy as np

def SLPTesting(x_test, y_test, w0, w1, w2):
    x0=1  # Bias term
    accuracy = 0
    for i in range(len(x_test)):
        x1 = x_test[i][0]
        x2 = x_test[i][1]
        y = y_test[i]
        # calculate the output of the neuron
        net = w0 * x0 + w1 * x1 + w2 * x2
        y_pred = np.sign(net)
        if y_pred == y:
            accuracy += 1
            
    return accuracy / len(x_test) * 100  # Return accuracy as a percentage        


        