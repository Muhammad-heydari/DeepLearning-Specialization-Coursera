import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from cat_recognizer_utils import *
from dnn_app_utils_v3 import load_data,print_mislabeled_images
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)



train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Example of a picture
# index = 25
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
# plt.imshow(train_x_orig[index])
# plt.show()

# information of my dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255
test_x = test_x_flatten/255
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    np.random.seed(1)
    n_x,n_h,n_y= layers_dims 
    costs = [] 
    m = X.shape[1]
    parameters = initialize_parameters(n_x=n_x,n_h=n_h,n_y=n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    grads = {}
    for i in range(0,num_iterations):
        A1,cacheL1 = linear_activation_forward(X,W1,b1,"relu")
        A2,cacheL2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        cost = compute_cost(A2,Y)
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2= linear_activation_backward(dA2,cacheL2,"sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1,cacheL1,"relu")
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters




def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    np.random.seed(1)
    costs = [] 
    parameters = initialize_parameters_deep(layers_dims)
    grads = {}
    for i in range(0,num_iterations):
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads= L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

##test my own image
def predictImage(imagePath):
    fileImage = Image.open(imagePath).convert("RGB").resize([num_px,num_px],Image.LANCZOS)
    my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
    image = np.array(fileImage)
    my_image = image.reshape(num_px*num_px*3,1)
    my_image = my_image/255.
    my_predicted_image = predict(my_image, my_label_y, parameters)
    plt.imshow(image)
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


## 2 layer model
# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 3500, print_cost=True)
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

## n layer model
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 3000, print_cost = True)
pred_train = predict_with_accuracy(train_x, train_y, parameters)
pred_test = predict_with_accuracy(test_x, test_y, parameters)

##get missed image
# print_mislabeled_images( classes,test_x, test_y, pred_test)

#predict own image 
predictImage("test1.jpg")
predictImage("test2.png")
predictImage("test3.jpg")
predictImage("test4.jpg")
