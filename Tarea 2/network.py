import random
import numpy as np


class QuadraticCost(object):
    
    @staticmethod
    def fn(a, y):
        """Retorna el costo cuadrático (MSE)."""
        return 0.5 * np.linalg.norm(a - y)**2
    
    @staticmethod
    def delta(z, a, y):
        """Retorna el delta para la última capa con costo cuadrático."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    
    @staticmethod
    def fn(a, y):
        """Retorna el costo de cross-entropy."""
        eps = 1e-12
        # Clipping para evitar log(0)
        a = np.clip(a, eps, 1.0 - eps)
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def delta(z, a, y):
        """Retorna el delta para la última capa con cross-entropy.
        Con cross-entropy y sigmoid, la derivada se simplifica a (a - y)."""
        return (a - y)


class Network(object):

    def __init__(self, sizes, cost=QuadraticCost, init="default", 
                 optimizer="sgd", beta1=0.9, beta2=0.999, eps=1e-8):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Inicialización de pesos y biases según el método seleccionado
        if init == "default":
            # Inicialización original: randn para pesos y biases
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        elif init == "xavier":
            # Inicialización Xavier (Glorot) para sigmoid/tanh
            self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) / np.sqrt(x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        elif init == "he":
            # Inicialización He para ReLU
            self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            raise ValueError(f"Método de inicialización '{init}' no reconocido. Use 'default', 'xavier' o 'he'.")

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)
        
        # Inicializar estados de Adam si es necesario
        if self.optimizer == "adam":
            if not hasattr(self, 'm_w'):
                self.m_w = [np.zeros(w.shape) for w in self.weights]
                self.v_w = [np.zeros(w.shape) for w in self.weights]
                self.m_b = [np.zeros(b.shape) for b in self.biases]
                self.v_b = [np.zeros(b.shape) for b in self.biases]
                self.t = 0
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        batch_size = len(mini_batch)
        
        if self.optimizer == "sgd":
            # Actualización SGD estándar
            self.weights = [w-(eta/batch_size)*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/batch_size)*nb
                           for b, nb in zip(self.biases, nabla_b)]
        
        elif self.optimizer == "adam":
            # Incrementar contador de pasos
            self.t += 1
            
            # Actualización Adam
            for i in range(len(self.weights)):
                # Gradientes promedio
                g_w = nabla_w[i] / batch_size
                g_b = nabla_b[i] / batch_size
                
                # Actualizar momentos
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * g_w
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (g_w ** 2)
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * g_b
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (g_b ** 2)
                
                # Corrección de bias
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                
                # Actualizar pesos y biases
                self.weights[i] -= eta * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
                self.biases[i] -= eta * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Delta en la última capa usando el método del costo
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))