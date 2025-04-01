#A01736353 Alejandro Daniel Moctezuma Cruz

## LIBRERIAS
import csv
import numpy as np
import pandas as pd

## FUNCIONES
def open_data(data):
    diabetes_data = pd.read_csv(data)
    return diabetes_data

def normalize_data(data):
    #Extract numeric values
    columns = data.iloc[:, :-1]

    #Mean and Standard Deviation
    mean_data = columns.mean()
    std_data = columns.std()
    
    data_normalized = (columns - mean_data) / std_data
    return data_normalized

def euclidean_distance(value_1, value_2):
    return np.sqrt(np.sum((value_1 - value_2) ** 2))

def knn(X, y, current_instance, k):
    distance_list = []

    #Calcular la distancia desde todos los puntos hacia punto actual
    for i in range(len(X)):
        dist = euclidean_distance(current_instance, X[i])
        distance_list.append((dist, y[i]))

    #Ordenar por distancia, encontrar k cercanos y separarlos
    distance_list.sort(key=lambda x: x[0])
    neighbors = distance_list[:k]
    class_count = {'tested_negative': 0, 'tested_positive': 0}
    for neighbor in neighbors:
        class_count[neighbor[1]] += 1
    return class_count

def generate_csv(data):
    #TODO generate either a .txt or .pdf
    pass

def algorithm():
    print("Clasificador Knn")
    print("A01736353 Alejandro Daniel Moctezuma Cruz")
    
    k_value = int(input("Introduce el valor de k: "))

    read_data = open_data('Data/Diabetes-Clasificacion.csv')
    normalized_data = normalize_data(read_data)

    #Variables dependientes e independientes 
    X = normalized_data.values
    y = read_data['class'].values

    #Iterar variable dependiente y determinar grupo
    output_results = []
    for i in range(len(X)):
        class_count = knn(X, y, X[i], k_value)

        positive_count = class_count['tested_negative']
        negative_count = class_count['tested_negative']

        if negative_count > positive_count:   
            corresponding_group = 'tested_negative'
        else:                       
            corresponding_group = 'tested_positive'

        output_results.append([i + 1, negative_count, positive_count, corresponding_group])

    print("¡El archivo se ha generado correctamente!")

#Main Execution
algorithm()