import mnist_loader
from network import Network, CrossEntropyCost
import pickle

# Cargar datos MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

# Crear y entrenar la red neuronal
# Configuración recomendada: CrossEntropy + Xavier + Adam
net = Network([784, 30, 10], cost=CrossEntropyCost, init="xavier", optimizer="adam")

# Entrenar con learning rate 0.001 (típico para Adam)
net.SGD(training_data, 15, 10, 0.001, test_data=test_data)

# Guardar la red entrenada
with open("red_entrenada.pkl", 'wb') as archivo:
    pickle.dump(net, archivo)