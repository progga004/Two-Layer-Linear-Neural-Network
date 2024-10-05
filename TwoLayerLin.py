import numpy as np


class Module():
    def __init__(self): 
        pass
    def forward(self,s_in):
        pass
    
    def backward(self, dL_ds_out, s_in):
        pass
    def update_var(self,dL_ds_out,s_in, stepsize):
        pass

    
    
class Linear(Module):
    def __init__(self, num_outparam, num_inparam, init_type='random'):
        if init_type == 'random':
            self.W = np.random.randn(num_outparam, num_inparam) * 0.01
            self.b = np.random.randn(num_outparam) * 0.01
        elif init_type == 'testcase':
            self.W = np.ones((num_outparam, num_inparam))
            self.b = np.ones((num_outparam))

    def forward(self, s_in):
       
       return np.dot(self.W,s_in) + np.outer(self.b,np.ones(s_in.shape[1]))
    

    def backward(self, dL_ds_out, s_in):
        self.dW = np.dot(dL_ds_out, s_in.T)  # Shape should be the same as self.W

        # Gradient with respect to biases
        self.db = np.sum(dL_ds_out, axis=1)  # Sum across samples if batch processing

        # Gradient with respect to input
        grad_input = np.dot(self.W.T, dL_ds_out)
        return grad_input
    def update_var(self, dL_ds_out, s_in, stepsize):
        dW = 0.
        db = 0.
        
   
        batchsize = s_in.shape[1]
        dW = np.dot(dL_ds_out,s_in.T)
        db = np.sum(dL_ds_out,axis=1)
        


        self.W = self.W - dW*stepsize
        self.b = self.b - db*stepsize
        
        return self.W,self.b

    
    #copy what you had from previous assignment
    
class ReLU(Module):
    def __init__(self):
        pass
    
    def forward(self,s_in):
        return np.maximum(0, s_in)
        
    
    def backward(self,dL_ds_out,s_in):
        dL_ds_in = dL_ds_out.copy()
        dL_ds_in[s_in <= 0] = 0
        return dL_ds_in
       
        
## multiclass classification
class Loss():
    pass
    

class MSELoss(Loss):
   # pass
    #copy what you had from previous assignment
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)/2

    def backward(self, y_pred, y_true):
        return  (y_pred - y_true) / y_true.size

 
    
class MNIST_2layer_binaryclass():
    def __init__(self, nhidden, istest=False):
        #self.layers = [Linear(100,784,0.01), ReLU(),   Linear(100,100,0.01),   ReLU(), Linear(10,100,0.01)]
        if istest:
            self.linear1 = Linear(nhidden,784, init_type = 'testcase')
        else:
            self.linear1 = Linear(nhidden,784, init_type = 'random')
        self.relu = ReLU()

        
        if istest:
            self.linear2 = Linear(1, nhidden, init_type = 'testcase')
        else:
            self.linear2 = Linear(1, nhidden, init_type = 'random')
            
        self.loss_fn = MSELoss()
        
        
    def forward(self, X, y):
        s1 = self.linear1.forward(X)
        h1 = self.relu.forward(s1)
        yhat = self.linear2.forward(h1)
        loss = self.loss_fn.forward(yhat, y)
        return loss, yhat

    def backward(self, X, y):
    # Forward pass to get output
       _, yhat = self.forward(X, y)
 
    # Compute the gradient of the loss function with respect to the network's output
       dL_dyhat = self.loss_fn.backward(yhat, y)

    # Backward pass through the second linear layer
       dL_dh1 = self.linear2.backward(dL_dyhat, self.relu.forward(self.linear1.forward(X)))

    # Backward pass through the ReLU activation
       dL_ds1 = self.relu.backward(dL_dh1, self.linear1.forward(X))

    # Backward pass through the first linear layer
       dL_dX = self.linear1.backward(dL_ds1, X)

    # Store the gradients (for use in update_params)
      

    def update_params(self, stepsize):
       self.linear1.W -= stepsize * self.linear1.dW
       self.linear1.b -= stepsize * self.linear1.db

        # Update the parameters of the second linear layer
       self.linear2.W -= stepsize * self.linear2.dW
       self.linear2.b -= stepsize * self.linear2.db
    def inference(self, X, y):
        loss, yhat = self.forward(X,y)
        yhat = np.sign(yhat)
        return loss, yhat
    # 
        