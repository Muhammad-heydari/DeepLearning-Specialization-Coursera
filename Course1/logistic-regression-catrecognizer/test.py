import numpy as np

a = np.array([[4,5],[6,7]])
b = np.array([[1,2],[3,4]])
print(a.shape)
dot = np.dot(a,b)
element_wise = np.multiply(a,b)
print(dot.shape)
print(dot)
print(element_wise.shape)
print(element_wise)