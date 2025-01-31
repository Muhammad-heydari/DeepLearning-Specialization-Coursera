import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(1)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters  
##test initializing
# parameters = initialize_parameters(3,2,1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W'+str(l)] =  np.random.randn(layer_dims[l],layer_dims[l-1])/ np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b'+str(l)] =  np.zeros((layer_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

##test inizializing deeply
# parameters = initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def linear_forward(A,W,b):
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache
##test lineraForward
# A, W, b = linear_forward_test_case()
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))


def linear_activation_forward(A_prev, W, b, activation):
    z,linear_cache = linear_forward(A_prev,W,b)
    cache = None
    if(activation == "sigmoid"):
        A,activation_cache = sigmoid(z)
        cache = (linear_cache, activation_cache)
    elif(activation == "relu"):
        A,activation_cache = relu(z)
        cache = (linear_cache, activation_cache)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    return A, cache
    
##test linear_activation_forward 
# A_prev, W, b = linear_activation_forward_test_case()
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))


def L_model_forward(X, parameters):
    caches = []
    L = len(parameters)//2
    A = X
    for l in range(1,L):
        A_prev = A
        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]
        A,linear_activation_cache = linear_activation_forward(A_prev,Wl,bl,"relu")
        caches.append(linear_activation_cache)
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    AL,linear_activation_cache = linear_activation_forward(A,WL,bL,"sigmoid")
    caches.append(linear_activation_cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL,caches
#test forward
# X, parameters = L_model_forward_test_case_2hidden()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost
##test compute cost
# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))


def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m=A_prev.shape[1]
    dW = (1/m) * np.dot(dZ,A_prev.T)
    db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev,dW,db

# test liniear backup
# dZ, linear_cache = linear_backward_test_case()
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    if(activation == "relu"):
        dZ = relu_backward(dA,activation_cache)
    elif(activation == "sigmoid"):
        dZ = sigmoid_backward(dA,activation_cache)
    dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db
##test liniear_activation_backward
# dAL, linear_activation_cache = linear_activation_backward_test_case()
# dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
# dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

def L_model_backward(AL, Y, caches):
    dA=None
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y,AL)- np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    dA_prev,dWL,dbL = linear_activation_backward(dAL,current_cache,"sigmoid")
    grads["dW"+str(L)] = dWL
    grads["db"+str(L)] = dbL
    grads["dA"+str(L-1)] = dA_prev
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev,dW,db = linear_activation_backward(grads['dA'+str(l+1)],current_cache,"relu")
        grads["dA"+str(l)] = dA_prev
        grads["dW"+str(l+1)] = dW
        grads["db"+str(l+1)] = db
    return grads
    
##test backward model
# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print_grads(grads)

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate*grads["dW"+str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate*grads["db"+str(l)])
    return parameters
##test update parameters
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)

# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))
def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
        
    return p

def predict_with_accuracy(X, y, parameters):
    m = X.shape[1]
    p = predict(X, y, parameters)
    print("Accuracy: "  + str(np.sum((p == y)/m)))
    return p
    



