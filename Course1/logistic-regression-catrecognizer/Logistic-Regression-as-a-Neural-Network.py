import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import scipy
from lr_utils import load_dataset

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[0].shape[0]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x =train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b=0
    assert(w.shape == (dim,1))
    return w,b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # FORWARD PROPAGATION 
    y_hat = sigmoid(np.dot(w.T,X) + b)              # compute activation
    cost = np.sum(((-np.log(y_hat))*Y + (-np.log(1-y_hat))*(1-Y)))/m  # compute cost

    # BACKWARD PROPAGATION 
    dw = (np.dot(X,(y_hat-Y).T))/m
    db = (np.sum(y_hat-Y))/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(X,Y,w,b,num_iteration,learning_rate):
    costs = []
    for i in range(num_iteration):
        grads,cost = propagate(w,b,X,Y)
        dw =  grads['dw']
        db = grads['db']
        w =w- learning_rate*dw
        b = b-learning_rate*db
        if  i%100 == 0:
            print("const at number " + str(i) + ": "+ str(cost))
            costs.append(cost)
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        if (A[0,i] >= 0.5):
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


def model(x_train,x_test,y_train,y_test,number_iteration,learning_rate):
    m = x_train.shape[0]
    W,B = initialize_with_zeros(m)
    params,grads,costs= optimize(x_train,y_train,W,B,number_iteration,learning_rate)
    train_predict = predict(params['w'],params["b"],x_train)
    test_predict = predict(params['w'],params["b"],x_test)
    print("train_accurity = " + str(100- np.mean(np.abs(train_predict-y_train)) * 100))
    print("test_accurity = "+str(100- np.mean(np.abs(test_predict-y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": test_predict, 
         "Y_prediction_train" : train_predict, 
         "w" : params['w'], 
         "b" : params["b"],
         "learning_rate" : learning_rate,
         "num_iterations": number_iteration}
    return d


d = model(train_set_x, test_set_x,train_set_y, test_set_y, 2000, 0.005)
print(d)


