import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import random

# Función para calcular distancia euclidiana manualmente
def euclidean(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Función para implementar SMOTE
def smote(X, y, minority_class, k=5, n_samples=100):
    # Filtrar datos de la clase minoritaria
    X_minority = X[y == minority_class]
    
    synthetic_samples = []
    for _ in range(n_samples):
        # Elegir un punto minoritario aleatorio
        i = random.randint(0, len(X_minority) - 1)
        point = X_minority[i]
        
        # Encontrar los k vecinos más cercanos dentro de la clase minoritaria
        distances = [euclidean(point, other) for other in X_minority]
        neighbors_idx = np.argsort(distances)[1:k + 1]  # Omitir el propio punto
        
        # Elegir un vecino aleatorio y generar un nuevo punto
        neighbor = X_minority[random.choice(neighbors_idx)]
        diff = neighbor - point
        new_sample = point + random.random() * diff
        synthetic_samples.append(new_sample)
    
    # Concatenar muestras sintéticas con las originales
    X_synthetic = np.vstack([X, np.array(synthetic_samples)])
    y_synthetic = np.hstack([y, [minority_class] * n_samples])
    
    return X_synthetic, y_synthetic

# Cargar el dataset "Glass"
X, y = fetch_openml(name="glass", version=1, as_frame=False, return_X_y=True)

# Convertir etiquetas categóricas a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Etiquetas numéricas

# División Hold-Out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Identificar clase minoritaria
class_counts = Counter(y_train)
minority_class = min(class_counts, key=class_counts.get)

# Función para clasificador Euclidiano
def euclidean_classifier(X_train, y_train, X_test):
    predictions = []
    for test_sample in X_test:
        distances = [euclidean(test_sample, train_sample) for train_sample in X_train]
        min_index = np.argmin(distances)
        predictions.append(y_train[min_index])
    return np.array(predictions)

# Clasificador 1NN con scikit-learn
knn = KNeighborsClassifier(n_neighbors=1)

# Validación Hold-Out antes de SMOTE
y_pred_euclid = euclidean_classifier(X_train, y_train, X_test)
y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
print("Antes de SMOTE (Hold-Out):")
print("Euclidiano Accuracy:", accuracy_score(y_test, y_pred_euclid))
print("1NN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Aplicar SMOTE al conjunto de entrenamiento
X_train_smote, y_train_smote = smote(X_train, y_train, minority_class, k=5, n_samples=200)

# Validación Hold-Out después de SMOTE
y_pred_euclid_smote = euclidean_classifier(X_train_smote, y_train_smote, X_test)
y_pred_knn_smote = knn.fit(X_train_smote, y_train_smote).predict(X_test)
print("\nDespués de SMOTE (Hold-Out):")
print("Euclidiano Accuracy:", accuracy_score(y_test, y_pred_euclid_smote))
print("1NN Accuracy:", accuracy_score(y_test, y_pred_knn_smote))

# Validación cruzada 10-Fold antes de SMOTE
print("\nAntes de SMOTE (10-Fold Cross-Validation):")
scores_knn = cross_val_score(knn, X, y, cv=10)
print("1NN Accuracy (promedio):", np.mean(scores_knn))

# Validación cruzada 10-Fold después de SMOTE
print("\nDespués de SMOTE (10-Fold Cross-Validation):")
scores_knn_smote = cross_val_score(knn, X_train_smote, y_train_smote, cv=10)
print("1NN Accuracy (promedio):", np.mean(scores_knn_smote))