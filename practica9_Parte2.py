import random
import numpy as np
from sklearn.datasets import load_iris

# Cargar el dataset Iris
data = load_iris()
X, y = data.data, data.target

# Filtrar las clases Setosa (0) y Virginica (2)
mask = (y == 0) | (y == 2)
X, y = X[mask], y[mask]

# Convertir las etiquetas a -1 y 1 para el perceptrón
y = np.where(y == 0, -1, 1)

# Dividir los datos en entrenamiento y prueba (Hold-Out 70/30)
def train_test_split(X, y, test_size=0.3):
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Implementar el perceptrón simple
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                # Actualizar pesos si hay error
                if y[idx] * linear_output <= 0:
                    self.weights += self.learning_rate * y[idx] * x_i
                    self.bias += self.learning_rate * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Entrenar el perceptrón
perceptron = Perceptron(learning_rate=0.01, n_iter=1000)
perceptron.fit(X_train, y_train)

# Validar en los datos de prueba
y_pred = perceptron.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print("Precisión en el conjunto de prueba:", accuracy)
