#A01736353 Alejandro Daniel Moctezuma Cruz

## LIBRARIES
import csv
import numpy as np
import pandas as pd

## FUNCTIONS
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

def generate_csv(data):
    #TODO generate either a .txt or .pdf
    pass

def algorithm():
    print("Clasificador Knn")
    print("A01736353 Alejandro Daniel Moctezuma Cruz")
    
    read_data = open_data('Data/Diabetes-Clasificacion.csv')
    normalized_data = normalize_data(read_data)
    #print(normalized_data)

    k_value = int(input("Introduce el valor de k: "))


#Main Execution
algorithm()