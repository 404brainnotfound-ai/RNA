"""
mnist_loader
~~~~~~~~~~~~
Una librería para cargar los datos de imágenes MNIST.  
Para más detalles sobre las estructuras de datos que se devuelven, 
mira los docstrings de ``load_data`` y ``load_data_wrapper``.  
En la práctica, ``load_data_wrapper`` es la función que usualmente 
se llama desde nuestro código de redes neuronales.
"""

#### Librerías
# Librería estándar
import pickle
import gzip

# Librerías de terceros
import numpy as np

def load_data():
    """Devuelve los datos de MNIST como una tupla que contiene los 
    datos de entrenamiento, los datos de validación y los datos de prueba.

    ``training_data`` se devuelve como una tupla con dos entradas.
    La primera entrada contiene las imágenes de entrenamiento reales.  
    Es un numpy ndarray con 50,000 elementos.  
    Cada elemento es, a su vez, un numpy ndarray con 784 valores, 
    que representan los 28 * 28 = 784 píxeles de una sola imagen MNIST.

    La segunda entrada en la tupla ``training_data`` es un numpy ndarray 
    que contiene 50,000 elementos.  Esos elementos son simplemente los 
    valores de los dígitos (0...9) correspondientes a las imágenes 
    contenidas en la primera entrada de la tupla.

    Los ``validation_data`` y ``test_data`` son similares, excepto que 
    cada uno contiene solo 10,000 imágenes.

    Este es un formato de datos conveniente, pero para usarlo en redes 
    neuronales es útil modificarlo un poco.  
    Eso se hace en la función envoltura ``load_data_wrapper()``, 
    ver más abajo.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Devuelve una tupla que contiene ``(training_data, validation_data,
    test_data)``.  Basada en ``load_data``, pero el formato es más
    conveniente para nuestra implementación de redes neuronales.

    En particular, ``training_data`` es una lista que contiene 50,000
    tuplas ``(x, y)``.  ``x`` es un numpy.ndarray de 784 dimensiones 
    que contiene la imagen de entrada.  ``y`` es un numpy.ndarray de 
    10 dimensiones que representa el vector unitario correspondiente 
    al dígito correcto para ``x``.

    ``validation_data`` y ``test_data`` son listas que contienen 10,000
    tuplas ``(x, y)``.  En cada caso, ``x`` es un numpy.ndarray de 784 
    dimensiones que contiene la imagen de entrada, y ``y`` es la 
    clasificación correspondiente, es decir, el valor del dígito 
    (entero) correspondiente a ``x``.

    Obviamente, esto significa que estamos usando formatos ligeramente 
    diferentes para los datos de entrenamiento y para los de validación/prueba.  
    Estos formatos resultan ser los más convenientes para usar en 
    nuestro código de redes neuronales.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Devuelve un vector unitario de 10 dimensiones con un 1.0 en la 
    posición j y ceros en el resto.  
    Esto se usa para convertir un dígito (0...9) en la salida deseada 
    correspondiente para la red neuronal.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
