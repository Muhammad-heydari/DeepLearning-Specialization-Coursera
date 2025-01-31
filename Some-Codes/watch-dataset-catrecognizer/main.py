from lr_utils import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import h5py

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print("train shape X:" + str(train_set_x_orig.shape))
print("train shape Y:" + str(train_set_y.shape))    
for i in range(len(train_set_x_orig[0])):
    plt.imshow(train_set_x_orig[i])
    print ("y = " + str(train_set_y[0, i]) + ", it's a '" + classes[np.squeeze(train_set_y[0, i])].decode("utf-8") +  "' picture.")
    plt.show()
print("test shape X:" + str(test_set_x_orig.shape))
print("test shape Y:" + str(test_set_y.shape))    
for i in range(len(test_set_x_orig[0])):
    plt.imshow(test_set_x_orig[i])
    print ("y = " + str(test_set_y[0, i]) + ", it's a '" + classes[np.squeeze(test_set_y[0, i])].decode("utf-8") +  "' picture.")
    plt.show()