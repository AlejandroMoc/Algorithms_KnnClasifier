#A01736353 Alejandro Daniel Moctezuma Cruz

## LIBRARIES
import csv
import numpy as np
import pandas as pd

## FUNCTIONS

def algorithm():
    pass

def open_data(data):
    with open(data, mode ='r')as file:
        diabetes_data = csv.reader(file)
        for lines in diabetes_data:
            print(lines)

def euclidean_distance():
    pass

def generate_output():
    #TODO generate either a .txt or .pdf
    pass

def main():
    print("Clasificador Knn")
    print("A01736353 Alejandro Daniel Moctezuma Cruz")
    
    open_data('Data/Diabetes-Clasificacion.csv')
    algorithm()


#Main Execution
main()