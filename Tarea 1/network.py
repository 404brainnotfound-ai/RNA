"""
network.py
~~~~~~~~~~
Un módulo para implementar el algoritmo de aprendizaje de descenso de gradiente estocástico en una red neuronal feedforward.
Los gradientes se calculan utilizando retropropagación.

Cabe señalar que me he centrado en hacer que el código sea simple, fácil de leer y fácil de modificar.
No está optimizado y omite muchas características deseables.

"""
# Librerías
import random
import numpy as np



class Network(object):

    def __init__(self, sizes):
        """
        Aquí le decimos a la red cuántas capas tendrá y cuántas neuronas
        habrá en cada capa.

            Ejemplo: [784, 30, 10]

                - 784: neuronas de entrada (por ejemplo, los píxeles de una imagen 28x28).
                - 30: neuronas en la capa escondida (donde la red empieza a aprender patrones).
                - 10: neuronas de salida (una para cada número del 0 al 9).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Aquí es donde la red toma un dato de entrada a y lo pasa por la capa que umltiplica la entrada por
        los pesos, le suma los sesgos. Luego aplica la función sigmoide para que el resultado quede entre 0 y 1
        y repite hasta llegar a la capa de salida.
        
        """""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        # training_data: lista con pares (entrada, respuesta correcta).
        # epochs: cuántas veces repasamos todos los datos.
        # mini_batch_size: cuántos ejemplos se usan en cada grupo pequeño.
        # eta: tasa de aprendizaje, define qué tan rápido aprende la red.
        # test_data: datos de prueba para medir si la red mejora.

        """
        Aquí entrenamos la red usando un método llamado "descenso de gradiente estocástico".
        La idea es que la red aprende poco a poco, viendo grupos pequeños de datos
        (mini-batches) en lugar de todos los datos de golpe.
        """

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Época {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Época {0} completada".format(j))

    def update_mini_batch(self, mini_batch, eta):

        """
        Toma un mini-batch y actualiza los pesos y sesgos de la red. Entonces 
        calcular cuánto se equivocó en cada ejemplo y luego promedia esos errores
        para hacer un solo ajuste.
        """
        
        # Inicializamos gradientes en ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Sumamos los errores  de cada ejemplo en el mini_batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    # RETROPROPAGACIÓN (BACKPROP)


    def backprop(self, x, y):
        """
        Aquí es donde la red "aprende del error".
        Calcula cuánto se equivocó en la salida y se reparte ese error
        hacia atrás, capa por capa, para saber cómo deben ajustarse
        los pesos y los sesgos.
        """


        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        # propagación hacia adelante
        activation = x
        activations = [x] # lista para almacenar todas las activaciones, capa por capa
        zs = [] # lista para almacenar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # retropropagación
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Nótese que la variable l en el bucle siguiente se usa un poco
        # diferente a la notación del Capítulo 2 del libro. Aquí,
        # l = 1 significa la última capa de neuronas, l = 2 es la penúltima capa,
        # y así sucesivamente. Es una renumeración del esquema en el libro,
        # usada aquí para aprovechar el hecho de que Python permite
        # índices negativos en listas.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Medimos cuántos ejemplos de prueba la red predice correctamente.
        La predicción final es la neurona con la mayor activación en la salida.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Calcula la diferencia entre lo que predijo la red y lo que debería ser.
        Es decir, cuánto nos equivocamos en la salida.
        """
        return (output_activations-y)

#### Funciones misceláneas
def sigmoid(z):
    """La función sigmoide que convierte cualquier número en un valor entre 0 y 1."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))