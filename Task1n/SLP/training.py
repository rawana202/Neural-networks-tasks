#1 random - global weights & bias
#2 loop for each epoch
# 3 loop for each sample 
# 4 calculate the output of the neuron
# 5 calculate the error
# 6 update the weights and bias

import numpy as np

def SLPTraining(x_train, y_train, lr, epochs, bias):

    w0,w1,w2=np.random.randn(3) * 0.01

    if bias == "Yes":
        x0=1
    else:
        x0=0

    for i in range(epochs):

        for j in range(len(x_train)):

            x1=x_train[j][0]
            x2=x_train[j][1]
            y=y_train[j]

            # calculate the output of the neuron
            net=w0*x0+w1*x1+w2*x2
            
            y_pred=np.sign(net)

            if y * y_pred <= 0:  
                # calculate the error
                error=y-y_pred
                # update the weights and bias
                w0=w0+lr*error*x0
                w1=w1+lr*error*x1
                w2=w2+lr*error*x2

    return w0, w1, w2        



