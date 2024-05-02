# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Función para cargar y preparar los datos desde archivos CSV
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Función para convertir variables continuas en categóricas binarias
def convert_to_binary_category(y):
    return (y > y.mean()).astype(int)

# Función para preprocesar los datos (escalar características)
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Función para entrenar y evaluar modelos
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calcular True Negatives (TN)
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    # Calcular False Positives (FP)
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    # Calcular Specificity
    specificity = tn / (tn + fp)
    
    return accuracy, precision, recall, f1, specificity

# Función para imprimir los resultados
def print_results(model_name, accuracy, precision, recall, f1, specificity):
    print("[" + model_name +"]"+ ":")
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Specificity: {:.2f}%".format(specificity * 100))
    print()

# Cargar y preparar el dataset Swedish Auto Insurance -------
auto_insurance_file = "swedish_auto_insurance.csv"
auto_insurance_data = load_data_from_csv(auto_insurance_file)
X_auto_insurance = auto_insurance_data.iloc[:, :-1]
y_auto_insurance = auto_insurance_data.iloc[:, -1]

# Convertir los valores continuos en categorías binarias
y_auto_insurance = convert_to_binary_category(y_auto_insurance)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(X_auto_insurance, y_auto_insurance, test_size=0.2, random_state=42)

# Preprocesar los datos
X_train_auto_scaled, X_test_auto_scaled = preprocess_data(X_train_auto, X_test_auto)

# Entrenar y evaluar modelos para el dataset Swedish Auto Insurance
print(" Resultados para Swedish Auto Insurance Dataset:\n")
models = [
    ("Regresion Logistica", LogisticRegression()),
    ("K-Vecinos Cercanos", KNeighborsClassifier()),
    ("Maquinas de Vectores de Soporte (SVM)", SVC(kernel='linear')),
    ("Naive Bayes", GaussianNB()),
    ("Red Neuronal", MLPClassifier(hidden_layer_sizes=(40, 50, 40), max_iter=1000))
]

for model_name, model in models:
    accuracy, precision, recall, f1, specificity = train_and_evaluate_model(model, X_train_auto_scaled, X_test_auto_scaled, y_train_auto, y_test_auto)
    print_results(model_name, accuracy, precision, recall, f1, specificity)

# Cargar y preparar el dataset Wine Quality ----------
wine_quality_file = "winequality-white.csv"
wine_quality_data = load_data_from_csv(wine_quality_file)
X_wine_quality = wine_quality_data.iloc[:, :-1]
y_wine_quality = wine_quality_data.iloc[:, -1]

# Convertir las etiquetas a un problema de clasificación binaria (calidad alta o baja)
y_wine_quality = (y_wine_quality > y_wine_quality.mean()).astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine_quality, y_wine_quality, test_size=0.2, random_state=42)

# Preprocesar los datos
X_train_wine_scaled, X_test_wine_scaled = preprocess_data(X_train_wine, X_test_wine)

# Entrenar y evaluar modelos para el dataset Wine Quality
print("\nResultados para Wine Quality Dataset:\n")
for model_name, model in models:
    accuracy, precision, recall, f1, specificity = train_and_evaluate_model(model, X_train_wine_scaled, X_test_wine_scaled, y_train_wine, y_test_wine)
    print_results(model_name, accuracy, precision, recall, f1, specificity)

# Cargar y preparar el dataset Pima Indians Diabetes ----------
pima_diabetes_file = "pima-indians-diabetes.csv"
pima_diabetes_data = load_data_from_csv(pima_diabetes_file)
X_pima = pima_diabetes_data.iloc[:, :-1]
y_pima = pima_diabetes_data.iloc[:, -1]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(X_pima, y_pima, test_size=0.2, random_state=42)

# Preprocesar los datos
X_train_pima_scaled, X_test_pima_scaled = preprocess_data(X_train_pima, X_test_pima)

# Entrenar y evaluar modelos para el dataset Pima Indians Diabetes
print("\nResultados para Pima Indians Diabetes Dataset:\n")
for model_name, model in models:
    accuracy, precision, recall, f1, specificity = train_and_evaluate_model(model, X_train_pima_scaled, X_test_pima_scaled, y_train_pima, y_test_pima)
    print_results(model_name, accuracy, precision, recall, f1, specificity)