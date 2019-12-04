import numpy as np
#test
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

training_imputs = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,1,0],
	 	  				    [1,1,1,0,1,0,0,1,0,0,0,1,0,0],
						    [0,1,1,0,0,0,0,0,0,0,1,1,0,0],
						    [1,0,0,1,0,1,1,1,0,1,1,1,1,0],
						    [1,1,1,0,1,1,1,0,1,1,1,1,0,1],
						    [0,1,0,0,1,1,0,1,1,0,1,1,0,0],
						    [0,0,1,1,1,1,1,1,0,0,0,1,0,0],
						    [1,0,0,1,0,1,0,0,0,1,1,1,1,1]])

training_outputs = np.array([[1,1,1,1,1,0,0,0]]).T

np.random.seed(1)	

synaptic_weights = 2 * np.random.random((14,1)) - 1	

#print("Случайные инициализируещие веса:")		
#print(synaptic_weights)		   

# Метод обратного распространения
for i in range(100000):
	input_layer = training_imputs
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))

	err = training_outputs - outputs
	adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)) )

	synaptic_weights += adjustments

#print("Веса после обучения:")
#print(synaptic_weights)

print("Результат после обучения:")
print(outputs)

#Тест
new_inputs = np.array([0,1,1,1,1,0,1,1,1,0,0,1,1,1]) #Новая
output = sigmoid(np.dot(new_inputs, synaptic_weights))

print("Новая ситуация:")
print(output)
