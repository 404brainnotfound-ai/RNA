# Importamos las librerías necesarias
import mnist_loader   # Sirve para cargar el dataset MNIST (imágenes de dígitos escritos a mano)
import network as network   # Aquí está la clase que define la red neuronal
import pickle   # Lo usaremos para guardar el modelo entrenado en un archivo

training_data, validation_data , test_data = mnist_loader.load_data_wrapper() 
# training_data: se usa para entrenar la red
# validation_data: se usa para ajustar parámetros 
# test_data: sirve para medir qué tan bien funciona la red


training_data = list(training_data)
test_data = list(test_data)
# Convertimos los datos a listas, porque la clase Network los maneja mejor así


net = network.Network([784, 30, 10])
# 784 neuronas de entrada (una por cada pixel de la imagen de 28x28)
# 30 neuronas en la capa oculta
# 10 neuronas en la capa de salida (una por cada dígito del 0 al 9)

# Entrenamos la red con el algoritmo SGD (Stochastic Gradient Descent)
net.SGD(training_data, 15, 10, 0.9, test_data=test_data)
# 15 épocas (el número de veces que la red verá todos los datos de entrenamiento)
# Mini_batch de tamaño 10 (entrena en grupos pequeños de 10 ejemplos)
# Learning rate = 0.9 (qué tan rápido aprende la red)
# test_data: lo usamos para evaluar el rendimiento en cada época

# Guardamos la red entrenada en un archivo .pkl para poder reutilizarla después
archivo = open("red_prueba.pkl", 'wb')
pickle.dump(net, archivo)
archivo.close()
exit()
