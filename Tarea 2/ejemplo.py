import mnist_loader
from network import Network, CrossEntropyCost
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net = Network([784, 30, 10], cost=CrossEntropyCost)

net.SGD(training_data, 15, 10, 0.1, test_data=test_data)

archivo = open("red_prueba1.pkl", 'wb')
pickle.dump(net, archivo)
archivo.close()
exit()
#leer el archivo

archivo_lectura = open("red_prueba.pkl", 'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD(training_data, 15, 50, 0.5, test_data=test_data)

archivo = open("red_prueba.pkl", 'wb')
pickle.dump(net, archivo)
archivo.close()
exit()

#esquema de cómo usar la red:
imagen = leer_imagen("disco.jpg")
print(net.feedforward(imagen))