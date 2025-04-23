#A01736353 Alejandro Daniel Moctezuma Cruz

## LIBRARIES
import csv
import numpy as np
import pandas as pd

## FUNCTIONS
def open_data(data):
    return pd.read_csv(data)

def normalize_data(data):
    #Solo valores numéricos
    columns = data.iloc[:, :-1]
    
    #Promedio y Desv. estandar
    mean_data = columns.mean()
    std_data = columns.std()

    data_normalized = (columns - mean_data) / std_data
    return data_normalized

def euclidean_distance(value_1, value_2):
    return np.sqrt(np.sum((value_1 - value_2) ** 2))

def knn(X, y, current_instance, k):
    distance_list: list = []

    #Calcular la distancia desde todos los puntos hacia punto actual
    for i in range(len(X)):
        dist = euclidean_distance(current_instance, X[i])
        distance_list.append((dist, y[i]))

    #Ordenar por distancia, encontrar k cercanos y separarlos
    distance_list.sort(key = lambda x: x[0])
    neighbors = distance_list[:k]
    class_count = {'tested_negative': 0, 'tested_positive': 0}
    for neighbor in neighbors:
        class_count[neighbor[1]] += 1
    return class_count

def generate_csv(output_results):
    pd.DataFrame(output_results, columns=["Instancia", "tested_negative", "tested_positive", "Clase asignada"]).to_csv('conteo_vecinos.csv', index = False)

def algorithm():
    print("Knn Clasifier")    
    k_value: int = int(input("Introduce el valor de k: "))

    #Abrir y normalizar datos
    data_entrenamiento = open_data('Data/Diabetes-Training.csv')
    normalized_entrenamiento = normalize_data(data_entrenamiento)
    data_clasificacion = open_data('Data/Diabetes-Clasification.csv')
    normalized_clasificacion = normalize_data(data_clasificacion)

    #Variables dependientes e independientes
    X_train = normalized_entrenamiento.values
    y_train = data_entrenamiento['class'].values
    X_test = normalized_clasificacion.values
    y_test = data_clasificacion['class'].values

    #Iterar variable dependiente y determinar grupo
    correct_counts: int = 0
    output_results: list = []
    for i in range(len(X_test)):
        class_count = knn(X_train, y_train, X_test[i], k_value)

        positive_count = class_count['tested_positive']
        negative_count = class_count['tested_negative']

        if negative_count > positive_count:
            corresponding_group = 'tested_negative'
        else:
            corresponding_group = 'tested_positive'

        output_results.append([i + 1, negative_count, positive_count, corresponding_group])

        #Verificar si coincide
        if y_test[i] == corresponding_group:
            correct_counts += 1

    successes_percentage = (correct_counts / len(X_test)) * 100
    generate_csv(output_results)

    print("¡El archivo se ha generado correctamente!")
    print("Porcentaje de instancias bien acertadas: ", successes_percentage, "%")

#Main Execution
algorithm()